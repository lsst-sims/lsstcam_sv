import healpy as hp
import numpy as np
from astropy.time import Time
from rubin_scheduler.scheduler.utils import (
    CurrentAreaMap,
    generate_all_sky,
    make_rolling_footprints,
)
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import DEFAULT_NSIDE

__all__ = ("survey_footprint",)


def survey_footprint(
    survey_start_mjd: float,
    nside: int = DEFAULT_NSIDE,
    almanac: Almanac | None = None,
    survey_info: dict | None = None,
) -> dict:
    if survey_info is None:
        survey_info = {}
    survey_info["nside"] = nside
    survey_info["survey_start"] = Time(survey_start_mjd, format="mjd", scale="utc")
    if almanac is None:
        almanac = Almanac(mjd_start=survey_start_mjd)
    survey_info["almanac"] = almanac

    # Define survey footprint areas
    allsky = generate_all_sky(nside=nside)
    wide_area = np.where(
        (
            (np.abs(allsky["eclip_lat"]) < 10)
            & ((allsky["eclip_lon"] > 240) | (allsky["eclip_lon"] < 40))
        ),
        True,
        False,
    )
    lvk_area = np.where(
        (
            (allsky["dec"] < -22)
            & (allsky["dec"] > -55)
            & (np.abs(allsky["gal_lat"]) > 15)
            & (allsky["gal_lat"] < 0)
        ),
        True,
        False,
    )
    # exclue wide_area from lvk_area
    lvk_area = np.where(wide_area & lvk_area, False, lvk_area)
    # This is just for easier visualization
    allsky["map"] = np.where(wide_area, 1, np.nan)
    allsky["map"] = np.where(lvk_area, 0.9, allsky["map"])
    allsky["map"][lvk_area & wide_area] = 1.1
    survey_info["skymap"] = allsky

    # Turn this into a Footprints class to use with the scheduler -
    # Set up wide area using standard filter balance and labels
    sky = CurrentAreaMap(nside=nside)
    # Slightly boost NES visits in g band to help ensure templates
    footprints_hp_array, labels = sky.return_maps(
        nes_ratios={"g": 0.35, "r": 0.4, "i": 0.4, "z": 0.28}
    )
    # Keep footprint inside wide_area along ecliptic
    for b in "ugrizy":
        footprints_hp_array[b][~wide_area] = 0
    survey_info["fp_array"] = footprints_hp_array

    rolling_labels = ["lowdust", "virgo"]
    rolling_idx = np.where(np.isin(labels, rolling_labels))
    wfd_labels = ["lowdust", "virgo", "bulgy", "LMC_SMC", "euclid_overlap"]
    wfd_map = np.where(np.isin(labels, wfd_labels), footprints_hp_array["r"], 0)
    survey_info["wfd_map"] = wfd_map

    # Have to convert footprints_hp_array to dict for make_rolling_footprints
    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]
    # Use the Almanac to find the position of the sun at the start of survey
    sun_moon_info = almanac.get_sun_moon_positions(survey_start_mjd)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    footprints = make_rolling_footprints(
        fp_hp=footprints_hp,
        mjd_start=survey_start_mjd,
        sun_ra_start=sun_ra_start,
        nslice=2,
        scale=0.9,
        nside=nside,
        wfd_indx=rolling_idx,
        order_roll=1,
        n_cycles=3,
        uniform=True,
    )
    survey_info["Footprints"] = footprints

    # Set up lvk_area footprint
    lvk_footprints_hp_array = np.zeros(
        hp.nside2npix(nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
    )
    for b in "gi":
        lvk_footprints_hp_array[b][lvk_area] = 1
    survey_info["extra_templates_array"] = lvk_footprints_hp_array

    lvk_footprints_hp = {}
    for key in lvk_footprints_hp_array.dtype.names:
        lvk_footprints_hp[key] = lvk_footprints_hp_array[key]

    lvk_footprints = make_rolling_footprints(
        fp_hp=lvk_footprints_hp,
        mjd_start=survey_start_mjd,
        sun_ra_start=sun_ra_start,
        nslice=2,
        scale=0.9,
        nside=nside,
        wfd_indx=rolling_idx,
        order_roll=1,
        n_cycles=3,
        uniform=True,
    )
    survey_info["ExtraTemplate_Footprints"] = lvk_footprints

    return survey_info
