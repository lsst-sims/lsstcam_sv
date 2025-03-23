import numpy as np
import healpy as hp

from astropy.time import Time
import astropy
astropy.utils.iers.conf.iers_degraded_accuracy = 'ignore'

from rubin_scheduler.scheduler.utils import generate_all_sky, Footprint

from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import Site, angular_separation

from rubin_scheduler.scheduler.model_observatory import ModelObservatory, KinemModel
from rubin_scheduler.scheduler.utils import ObservationArray

from rubin_scheduler.scheduler.utils import SchemaConverter, run_info_table
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers


__all__ = ['survey_times', 'lst_over_survey_time',
           'survey_footprint', 'setup_observatory',
           'count_obstime', 'save_opsim']

def survey_times(survey_start: Time | None = None,
                 survey_end: Time | None = None,
                 verbose: bool = True) -> dict:
    if survey_start is None:
        survey_start = Time('2025-06-11T12:00:00', format='isot', scale='utc')
    if survey_end is None:
        survey_end = Time('2025-08-30T12:00:00', format='isot', scale='utc')
    survey_length = int(np.ceil((survey_end - survey_start).jd))

    if verbose:
        print("Survey start", survey_start.iso)
        print("Survey end", survey_end.isot)
        print("for a survey length of (nights)", survey_length)

    site = Site("LSST")
    almanac = Almanac(mjd_start=survey_start.mjd)
    match = np.where(
        (almanac.sunsets["night"] > 0) & (almanac.sunsets["night"] <= survey_length)
    )
    sunsets = almanac.sunsets[match]["sun_n12_setting"]
    sunrises = almanac.sunsets[match]["sun_n12_rising"]

    survey_info = {'survey_start': survey_start,
                   'survey_end': survey_end,
                   'survey_length': survey_length,
                   'sunsets': sunsets,
                   'sunrises': sunrises,
                   'almanac': almanac,
                   'site': site}

    # Add downtime and up/down calculation into survey_info
    # Let's assume for the first 30 days -
    # only the last 3 hours of the night are available
    step1 = 30
    down_starts = sunsets[0:step1]
    down_ends = sunrises[0:step1] - 3.0 / 24
    # And for the next 20 days, the first 3 hours of the night are unavailable
    step2 = 50
    down_starts = np.concatenate([down_starts, sunsets[step1:step2]])
    down_ends = np.concatenate([down_ends, sunsets[step1:step2] + 3.0 / 24])
    # And then from July through end of August, gets all of the night
    down_starts = np.concatenate([down_starts, sunsets[step2:]])
    down_ends = np.concatenate([down_ends, sunsets[step2:]])

    # Then let's add random periods of downtime within each night
    rng = np.random.default_rng(seed=55)
    # Assume some chance of having some amount of downtime
    # (but this is simplistic and assumes downtime once per night
    threshold = 1.0
    prob_down = rng.random(len(sunsets))
    time_down = rng.gumbel(loc=0.5, scale=1, size=len(sunsets))  # in hours
    # apply probability of having downtime or not
    time_down = np.where(prob_down <= threshold, time_down, 0)
    hours_in_night = (sunrises - sunsets) * 24
    avail_in_night = hours_in_night - (down_ends - down_starts) * 24
    time_down = np.where(time_down >= avail_in_night, avail_in_night - 0.1, time_down)
    time_down = np.where(time_down <= 0, 0, time_down)
    latest_start = avail_in_night - time_down
    d_starts = rng.uniform(low=down_ends, high=down_ends + latest_start / 24.0)
    d_ends = d_starts + time_down / 24.0

    # combine previous expected downtime and random downtime
    down_starts = np.concatenate([down_starts, d_starts])
    down_ends = np.concatenate([down_ends, d_ends])

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
    for start, end in zip(downtimes['start'], downtimes['end']):
        idx = np.where((start > dayobsmjd) & (end < dayobsmjd + 1))
        downtime_per_night[idx] += (end - start) * 24

    survey_info['step1'] = step1
    survey_info['step2'] = step2
    survey_info['downtimes'] = downtimes
    survey_info['dayobsmjd'] = dayobsmjd
    survey_info['hours_in_night'] = hours_in_night
    survey_info['downtime_per_night'] = downtime_per_night
    survey_info['avail_per_night'] = hours_in_night - downtime_per_night

    if verbose:
        print(f"Max length of night {hours_in_night.max()} min length of night {hours_in_night.min()}")
        print(f"Total nighttime {hours_in_night.sum()}, "
              f"total downtime {downtime_per_night.sum()}, "
              f"available time {hours_in_night.sum() - downtime_per_night.sum()}")

    return survey_info

def lst_over_survey_time(survey_info: dict) -> None:
    # Some informational stuff to help define the footprint
    loc = survey_info['site'].to_earth_location()
    idx = int(survey_info['survey_length'] / 2)
    sunsets = survey_info['sunsets']
    sunrises = survey_info['sunrises']
    mid_survey = Time(sunsets[idx] + (sunrises[idx] - sunsets[idx]) / 2, format='mjd', scale='utc', location=loc)
    mid_lst = mid_survey.sidereal_time('mean')
    idx = 0
    start_lst = Time(sunsets[idx] + (
                sunrises[idx] - sunsets[idx]) / 2, format='mjd', scale='utc', location=loc).sidereal_time('mean')
    idx = -1
    end_lst = Time(sunsets[idx] + (
                sunrises[idx] - sunsets[idx]) / 2, format='mjd', scale='utc', location=loc).sidereal_time('mean')

    print('lst midnight @ start', start_lst.deg, 'lst midnight @ mid', mid_lst.deg, 'lst midnight @ end', end_lst.deg)
    idx = int(survey_info['survey_length'] / 2)
    sunset_mid_lst = Time(sunsets[idx], format='mjd', scale='utc', location=loc).sidereal_time('mean')
    sunrise_mid_lst = Time(sunrises[idx], format='mjd', scale='utc', location=loc).sidereal_time('mean')
    print('lst sunset @ mid', sunset_mid_lst.deg, 'lst sunrise @ mid', sunrise_mid_lst.deg)


def survey_footprint(survey_info: dict, nside: int = 32, verbose=False):
    # Define survey footprint

    site = survey_info['site']
    survey_info['nside'] = nside

    sky = generate_all_sky(nside=nside)

    # Keith's suggested Wide areas - two regions ..
    south = np.where((((sky['ra'] > 280) & (sky['ra'] <= 360)) | ((sky['ra'] >= 0) & (sky['ra'] < 10)))
                     & (sky['dec'] < -40) & (sky['dec'] > -60))
    north = np.where((((sky['ra'] > 310) & (sky['ra'] <= 360)) | ((sky['ra'] >= 0) & (sky['ra'] < 10)))
                     & (sky['dec'] < 5) & (sky['dec'] > -5))

    sky['map'] = np.zeros(hp.nside2npix(nside)) + np.nan
    sky['map'][south] = 1
    sky['map'][north] = 1

    if verbose:
        # Check individually, relevant for blob coverage..
        area_south = len(np.where(sky['map'][south] == 1)[0]) * hp.nside2pixarea(nside, degrees=True)
        approx_nfields_south = area_south / 9.6 * 1.3
        print('south area', area_south, 'south approx n fields', approx_nfields_south)
        area_north = len(np.where(sky['map'][north] == 1)[0]) * hp.nside2pixarea(nside, degrees=True)
        approx_nfields_north = area_north / 9.6 * 1.3
        print('north area', area_north, 'south approx n fields', approx_nfields_north)
        # And together
        print('total area', area_south + area_north, 'total approx n fields', approx_nfields_south + approx_nfields_north)

        # How many fields are typically in a 'blob' with pair time of 33 minutes?
        print('approx blob fields', 33  * 60 / (32 + 8))
        # so .. looks like we'd cover the north and then south in a ~~two blobs of pairs each?

        print('closest approach to ecliptic in south area (deg)?', sky['eclip_lat'][south].min())
        print('max/min elevation in south?', abs(sky['dec'][south].max() - site.latitude),
            abs(sky['dec'][south].min() - survey_info['site'].latitude))
        print('max/min elevation in north?', abs(site.latitude - sky['dec'][north].min()),
            abs(site.latitude - sky['dec'][north].max()))

    # Deep field with good visibility
    dd_ra = 310
    dd_dec = -50
    radius = np.sqrt(100 / np.pi)
    dist = angular_separation(dd_ra, dd_dec, sky['ra'], sky['dec'])
    close = np.where(dist <= radius)[0]
    dd = np.zeros(hp.nside2npix(nside)) + np.nan
    dd[close] = 1

    survey_info['sky'] = sky
    survey_info['deep'] = {'ra': dd_ra, 'dec': dd_dec, 'radius': radius}

    if verbose:
        area_dd = len(np.where(dd == 1)[0]) * hp.nside2pixarea(nside, degrees=True)
        approx_nfields_dd = area_dd / 9.6 * 1.5
        print("DDF radius", radius, "DD area, not including dither edges",
            area_dd, "very approximate n fields", approx_nfields_dd)

        # Make a nice looking figure
        temp = np.where(np.isnan(sky['map']), 0, sky['map']) + np.where(np.isnan(dd), 0, 2 * dd)
        temp[np.where(temp == 0)] = np.nan
        temp = np.ma.MaskedArray(temp)
        s = slicers.HealpixSlicer(nside=nside)
        p = plots.HpxmapPlotter()
        fig = p(temp, s, {'color_max': 2, 'color_min': 0, 'n_ticks': 5})
        survey_info['figure'] = fig

    # Turn this into footprint_hp to use with the scheduler -

    # Wide SV - goal is 60/60/60/60 visits per pointing in griz
    band_ratios = {"u": 0, "g": 1.0, "r": 1.0, "i": 1.0, "z": 1.0, "y": 0}

    footprints_hp = np.zeros(hp.nside2npix(nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)), )
    for b in 'ugrizy':
        footprints_hp[b] = sky['map'] * band_ratios[b]
        footprints_hp[b][np.isnan(footprints_hp[b])] = 0

    if verbose:
        print('')
        wide_goals = [round(band_ratios[k] * 60) for k in band_ratios]
        print('wide goal visits per filter per point', wide_goals)

    # Deep SV - goal is ugrizy like year two of WFD
    # so use low-dust-ratios
    lowdust_band_ratios = {"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9}

    dd_footprints_hp = np.zeros(hp.nside2npix(nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)), )
    for b in 'ugrizy':
        dd_footprints_hp[b] = dd * lowdust_band_ratios[b] * 3
        dd_footprints_hp[b][np.isnan(dd_footprints_hp[b])] = 0

    if verbose:
        # estimates of number of visits per night / sequence for DDF (for field survey)
        dd_goals = [round(lowdust_band_ratios[k] * 184) for k in lowdust_band_ratios]
        print('deep goal visits per filter per point', dd_goals)
        print("deep visits per day, 30 day survey", np.array(dd_goals) / 30, np.array(dd_goals).sum() / 30)

        # This would likely be the necessary visits per night that are required? Single sequence quite long.
        seq = {'u': 16, 'g': 4, 'r': 12, 'i': 12, 'z': 12, 'y': 31}
        # Try shorter seq, run twice.
        dd_seq = {'u': 9, 'g': 2, 'r': 7, 'i': 8, 'z': 7, 'y': 17}
        tt = [round(seq[k] * 1.0) for k in seq]
        print("deep via sequence? approx ave nvis in sequence", (np.array(tt).sum() - 13) * 1.5,
            "approx time (min) of sequence", (np.array(tt) * (32 + 5)).sum() / 60)

    # combined Wide and Deep footprint
    combined_hp = np.zeros(hp.nside2npix(nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)), )
    for b in 'ugrizy':
        wide_dd_masked = np.where(dd_footprints_hp[b] > 0, 0, footprints_hp[b])
        combined_hp[b] = wide_dd_masked + dd_footprints_hp[b]

    survey_info['wide_hp'] = footprints_hp
    survey_info['deep_hp'] = dd_footprints_hp
    survey_info['combined_hp'] = combined_hp

    return survey_info

def setup_observatory(survey_info: dict)-> ModelObservatory:
    model_obs = ModelObservatory(nside=survey_info['nside'],
                                   mjd=survey_info['survey_start'].mjd,
                                   mjd_start=survey_info['survey_start'].mjd,
                                   cloud_data="ideal", # noclouds
                                   seeing_data=None, # standard seeing
                                   wind_data=None,
                                   downtimes = survey_info['downtimes'])
    # Slow the telescope down with smaller jerk/acceleration and maxvel
    # But is faster with settle of 0 (Tiago is working on this)
    model_obs.setup_telescope(
        altitude_maxspeed=2.0,
        altitude_accel=2.0,
        altitude_jerk=8.0,
        azimuth_maxspeed=2.0,
        azimuth_accel=2.0,
        azimuth_jerk=8.0,
        settle_time=0.0,
    )
    return model_obs


def count_obstime(observations: ObservationArray, survey_info: dict) -> dict:
    obs_time = np.zeros(len(survey_info['sunrises']))
    for i in range(len(survey_info['sunrises'])):
        idx = np.where((observations['mjd'] >= survey_info['sunsets'][i])
                       & (observations['mjd'] <= survey_info['sunrises'][i]))[0]
        obs_time[i] = (observations['visittime'][idx] + observations['slewtime'][idx]).sum() / 60 / 60

    tnight = float(survey_info['hours_in_night'].sum())
    tdown = float(survey_info['downtime_per_night'].sum())
    tavail = float(survey_info['avail_per_night'].sum())
    tobs = float((observations['visittime'].sum() + observations['slewtime'].sum()) / 60 / 60)

    print(f"Total night time (hours): {tnight:.2f}")
    print(f"Total down time (hours): {tdown:.2f}")
    print(f"Total available time (hours): {tavail:.2f}")
    print(f"Total time in observations + slew (hours): {tobs:.2f}")
    print(f"Unscheduled time (hours): {(tavail - tobs):.2f}")

    survey_info['obs_time_per_night'] = obs_time
    return survey_info

def save_opsim(filename: str, observatory: ModelObservatory, observations: ObservationArray) -> None:
    info = run_info_table(observatory)
    converter = SchemaConverter()
    converter.obs2opsim(observations, filename=filename, info=info, delete_past=True)
    visits_df = converter.obs2opsim(observations)
    return visits_df