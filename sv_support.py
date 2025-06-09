import astropy
import healpy as hp
import numpy as np
from astropy.time import Time

astropy.utils.iers.conf.iers_degraded_accuracy = "ignore"

import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers
from rubin_scheduler.scheduler.model_observatory import (KinemModel,
                                                         ModelObservatory,
                                                         tma_movement)
from rubin_scheduler.scheduler.utils import (ConstantFootprint, Footprint,
                                             ObservationArray, SchemaConverter,
                                             generate_all_sky,
                                             make_rolling_footprints,
                                             run_info_table)
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import Site, angular_separation

from custom_step import PerFilterStep

__all__ = [
    "survey_times",
    "lst_over_survey_time",
    "survey_footprint",
    "setup_observatory",
    "count_obstime",
    "save_opsim",
]


def survey_times(verbose: bool = True, random_seed: int = 55) -> dict:
    survey_start = Time("2025-06-15T12:00:00", format="isot", scale="utc")
    survey_end = Time("2025-09-15T12:00:00", format="isot", scale="utc")
    survey_length = int(np.ceil((survey_end - survey_start).jd))

    survey_mid = Time("2025-07-01T12:00:00", format="isot", scale="utc")

    if verbose:
        print("Survey start", survey_start.iso)
        print("Survey end", survey_end.isot)
        print("for a survey length of (nights)", survey_length)

    site = Site("LSST")
    almanac = Almanac(mjd_start=survey_start.mjd)
    alm_start = np.where(abs(almanac.sunsets["sunset"] - survey_start.mjd) < 0.5)[0][0]
    alm_end = np.where(abs(almanac.sunsets["sunset"] - survey_end.mjd) < 0.5)[0][0]
    alm_mid = np.where(abs(almanac.sunsets["sunset"] - survey_mid.mjd) < 0.5)[0][0]
    mid_offset = alm_mid - alm_start

    sunsets = almanac.sunsets[alm_start:alm_end]["sun_n12_setting"]
    civil_sunsets = almanac.sunsets[alm_start:alm_end]["sunset"]
    civil_sunrises = almanac.sunsets[alm_start:alm_end]["sunrise"]
    sunrises = almanac.sunsets[alm_start:alm_end]["sun_n12_rising"]
    moon_set = almanac.sunsets[alm_start:alm_end]["moonset"]
    moon_rise = almanac.sunsets[alm_start:alm_end]["moonrise"]
    moon_illum = almanac.interpolators["moon_phase"](sunsets)

    survey_info = {
        "survey_start": survey_start,
        "survey_end": survey_end,
        "survey_length": survey_length,
        "almanac": almanac,
        "sunsets": sunsets,
        "sunrises": sunrises,
        "moonsets": moon_set,
        "moonrises": moon_rise,
        "moon_illum": moon_illum,
        "almanac": almanac,
        "site": site,
        "nside": 32,
    }

    # Add time limits and downtime
    # SV survey never operates during last hour of the night
    # 2 hours per night for SV survey from June 15 - July 1
    # choose best time (moon down)
    # whole night from July 1 - September 15 with random downtime
    # 25% downtime ?

    rng = np.random.default_rng(seed=random_seed)

    night_start = sunsets
    night_end = sunrises - 1.0 / 24
    # Almanac always counts moon rise and set in the current night, after sunset
    # dark ends with either sunrise or moonrise
    dark_end = np.where(moon_rise < night_end, moon_rise, night_end)
    # dark starts with either sunset or moonset
    dark_start = np.where(moon_set < night_end, moon_set, night_start)
    # but sometimes the moon can be up the whole night
    up_all_night = np.where(dark_end < dark_start)
    dark_start[up_all_night] = 0
    dark_end[up_all_night] = 0

    two_hours = "random"
    if two_hours == "random":
        # Choose a random 2 hours within these times
        obs_start = np.zeros(len(dark_start))
        good = np.where(
            (dark_end - dark_start - 2.5 / 24 > 0) & (night_start < survey_mid.mjd)
        )
        obs_start[good] = rng.uniform(
            low=dark_start[good], high=dark_end[good] - 2.5 / 24
        )
        # Add downtime from the start of the night until the observations
        down_starts = civil_sunsets[good]
        down_ends = obs_start[good]
        # Add downtime from after the observation until sunrise
        down_starts = np.concatenate([down_starts, obs_start[good] + 2.1 / 24.0])
        down_ends = np.concatenate([down_ends, civil_sunrises[good]])
        # if moon up all night AND before survey_mid, remove the whole night
        bad = np.where(
            (dark_end - dark_start - 2.5 / 24 <= 0) & (night_start < survey_mid.mjd)
        )
        down_starts = np.concatenate([down_starts, civil_sunsets[bad]])
        down_ends = np.concatenate([down_ends, civil_sunrises[bad]])

    # Then let's add random periods of downtime within each night,
    # Assume some chance of having some amount of downtime
    # (but this is simplistic and assumes downtime ~twice per night)
    random_downtime = 0
    for count in range(3):
        threshold = 1.0 - (count / 5)
        prob_down = rng.random(len(night_start))
        time_down = rng.gumbel(loc=0.5, scale=1, size=len(night_start))  # in hours
        # apply probability of having downtime or not
        time_down = np.where(prob_down <= threshold, time_down, 0)
        avail_in_night = (night_end - night_start) * 24
        time_down = np.where(time_down >= avail_in_night, avail_in_night, time_down)
        time_down = np.where(time_down <= 0, 0, time_down)
        d_starts = rng.uniform(low=night_start, high=night_end - time_down / 24)
        d_ends = d_starts + time_down / 24.0
        # But only use these after July 15 -
        d_starts = d_starts[mid_offset:]
        d_ends = d_ends[mid_offset:]
        random_downtime += ((d_ends - d_starts) * 24).sum()
        night_hours = avail_in_night[mid_offset:].sum()
        print(random_downtime, night_hours, random_downtime / night_hours)
        # combine previous expected downtime and random downtime
        down_starts = np.concatenate([down_starts, d_starts])
        down_ends = np.concatenate([down_ends, d_ends])

    # And mask the final hour of the night through the survey
    # (already done for 0:mid)
    down_starts = np.concatenate([down_starts, sunrises[mid_offset:] - 1.0 / 24])
    down_ends = np.concatenate([down_ends, civil_sunrises[mid_offset:]])

    # Turn into an array of downtimes for sim_runner
    # down_starts/ down_ends should be mjd times for internal-ModelObservatory use
    downtimes = np.array(
        list(zip(down_starts, down_ends)),
        dtype=list(zip(["start", "end"], [float, float])),
    )
    downtimes.sort(order="start")

    # Eliminate overlaps (just in case)
    diff = downtimes["start"][1:] - downtimes["end"][0:-1]
    while np.min(diff) < 0:
        print("found overlap")
        # Should be able to do this without a loop, but this works
        for i, dt in enumerate(downtimes[0:-1]):
            if downtimes["start"][i + 1] < dt["end"]:
                new_end = np.max([dt["end"], downtimes["end"][i + 1]])
                downtimes[i]["end"] = new_end
                downtimes[i + 1]["end"] = new_end

        good = np.where(downtimes["end"] - np.roll(downtimes["end"], 1) != 0)
        downtimes = downtimes[good]
        diff = downtimes["start"][1:] - downtimes["end"][0:-1]

    # Count up downtime within each night
    dayobsmjd = np.arange(survey_start.mjd, survey_start.mjd + survey_length, 1)
    downtime_per_night = np.zeros(len(sunrises))
    for start, end in zip(downtimes["start"], downtimes["end"]):
        idx = np.where((start > dayobsmjd) & (end < dayobsmjd + 1))
        if start < sunsets[idx]:
            dstart = sunsets[idx]
        else:
            dstart = start
        if end > sunrises[idx]:
            dend = sunrises[idx]
        else:
            dend = end
        downtime_per_night[idx] += (dend - dstart) * 24

    survey_info["downtimes"] = downtimes
    survey_info["dayobsmjd"] = dayobsmjd
    hours_in_night = (sunrises - sunsets) * 24
    survey_info["hours_in_night"] = hours_in_night
    survey_info["downtime_per_night"] = downtime_per_night
    survey_info["avail_per_night"] = hours_in_night - downtime_per_night

    if verbose:
        print(
            f"Max length of night {hours_in_night.max()} min length of night {hours_in_night.min()}"
        )
        print(
            f"Total nighttime {hours_in_night.sum()}, "
            f"total downtime {downtime_per_night.sum()}, "
            f"available time {hours_in_night.sum() - downtime_per_night.sum()}"
        )

    return survey_info


def lst_over_survey_time(survey_info: dict) -> None:
    # Some informational stuff to help define the footprint
    loc = survey_info["site"].to_earth_location()
    idx = int(survey_info["survey_length"] / 2)
    sunsets = survey_info["sunsets"]
    sunrises = survey_info["sunrises"]
    mid_survey = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    )
    mid_lst = mid_survey.sidereal_time("mean")
    idx = 0
    start_lst = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    ).sidereal_time("mean")
    idx = -1
    end_lst = Time(
        sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2,
        format="mjd",
        scale="utc",
        location=loc,
    ).sidereal_time("mean")

    print(
        "lst midnight @ start",
        start_lst.deg,
        "lst midnight @ mid",
        mid_lst.deg,
        "lst midnight @ end",
        end_lst.deg,
    )
    idx = int(survey_info["survey_length"] / 2)
    sunset_mid_lst = Time(
        sunsets[idx], format="mjd", scale="utc", location=loc
    ).sidereal_time("mean")
    sunrise_mid_lst = Time(
        sunrises[idx], format="mjd", scale="utc", location=loc
    ).sidereal_time("mean")
    print(
        "lst sunset @ mid", sunset_mid_lst.deg, "lst sunrise @ mid", sunrise_mid_lst.deg
    )


def survey_footprint(survey_info: dict, verbose=False):
    # Define survey footprint

    site = survey_info["site"]
    nside = survey_info["nside"]
    sky = generate_all_sky(nside=nside)
    # use ecliptic
    sky["map"] = np.where(sky["map"] == 0, 1, np.nan)
    sky["map"] = np.where(np.abs(sky["eclip_lat"]) <= 10, sky["map"], np.nan)
    sky["map"] = np.where(
        (sky["eclip_lon"] > 250) | (sky["eclip_lon"] < 80), sky["map"], np.nan
    )

    survey_info["sky"] = sky

    # Turn this into footprint_hp to use with the scheduler -

    # low-dust wfd ratios
    low_dust_ratios = {"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9}

    footprints_hp = np.zeros(
        hp.nside2npix(nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
    )
    for b in "ugrizy":
        footprints_hp[b] = sky["map"] * low_dust_ratios[b]
        footprints_hp[b][np.isnan(footprints_hp[b])] = 0

    survey_info["footprint_hp"] = footprints_hp

    # For PerFilterStep, need to know bright/dark
    moon_illums = survey_info["moon_illum"]
    illum_limit = 40
    u_loaded = np.where(moon_illums <= illum_limit)[0]
    y_loaded = np.where(moon_illums > illum_limit)[0]

    mystep = PerFilterStep(
        survey_length=survey_info["survey_length"],
        loaded_dict={"u": u_loaded, "y": y_loaded},
    )
    footprints = Footprint(survey_info["survey_start"].mjd, step_func=mystep)
    for f in footprints_hp.dtype.names:
        footprints.set_footprint(f, footprints_hp[f])

    # # Use the Almanac to find the position of the sun at the start of survey
    # sun_moon_info = survey_info['almanac'].get_sun_moon_positions(mjd_start)
    # sun_ra_start = sun_moon_info["sun_RA"].copy()

    # footprints = make_rolling_footprints(
    #     fp_hp=footprints_hp,
    #     mjd_start=mjd_start,
    #     sun_ra_start=sun_ra_start,
    #     nslice=nslice,
    #     scale=rolling_scale,
    #     nside=nside,
    #     wfd_indx=wfd_indx,
    #     order_roll=1,
    #     n_cycles=3,
    #     uniform=rolling_uniform,
    # )

    survey_info["footprints_obj"] = footprints

    return survey_info


def setup_observatory(survey_info: dict) -> ModelObservatory:
    model_obs = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data="ideal",  # noclouds
        seeing_data=None,  # standard seeing
        wind_data=None,
        downtimes=survey_info["downtimes"],
    )
    # Slow the telescope down with smaller jerk/acceleration and maxvel
    tma = tma_movement(70)
    # But is faster with settle of 0 (Tiago is working on this)
    model_obs.setup_telescope(**tma)
    return model_obs


def setup_observatory_slow(survey_info: dict) -> ModelObservatory:
    model_obs = ModelObservatory(
        nside=survey_info["nside"],
        mjd=survey_info["survey_start"].mjd,
        mjd_start=survey_info["survey_start"].mjd,
        cloud_data="ideal",  # noclouds
        seeing_data=None,  # standard seeing
        wind_data=None,
        downtimes=survey_info["downtimes"],
    )
    # Slow the telescope down with smaller jerk/acceleration and maxvel
    tma = tma_movement(10)
    tma["settle_time"] = 6
    model_obs.setup_telescope(**tma)

    return model_obs


def count_obstime(observations: ObservationArray, survey_info: dict) -> dict:
    obs_time = np.zeros(len(survey_info["sunrises"]))
    for i in range(len(survey_info["sunrises"])):
        idx = np.where(
            (observations["mjd"] >= survey_info["sunsets"][i])
            & (observations["mjd"] <= survey_info["sunrises"][i])
        )[0]
        obs_time[i] = (
            (observations["visittime"][idx] + observations["slewtime"][idx]).sum()
            / 60
            / 60
        )

    tnight = float(survey_info["hours_in_night"].sum())
    tdown = float(survey_info["downtime_per_night"].sum())
    tavail = float(survey_info["avail_per_night"].sum())
    tobs = float(
        (observations["visittime"].sum() + observations["slewtime"].sum()) / 60 / 60
    )

    print(f"Total night time (hours): {tnight:.2f}")
    print(f"Total down time (hours): {tdown:.2f}")
    print(f"Total available time (hours): {tavail:.2f}")
    print(f"Total time in observations + slew (hours): {tobs:.2f}")
    print(f"Unscheduled time (hours): {(tavail - tobs):.2f}")

    survey_info["obs_time_per_night"] = obs_time
    return survey_info


def save_opsim(
    filename: str, observatory: ModelObservatory, observations: ObservationArray
) -> None:
    info = run_info_table(observatory)
    converter = SchemaConverter()
    converter.obs2opsim(observations, filename=filename, info=info, delete_past=True)
    visits_df = converter.obs2opsim(observations)
    return visits_df
