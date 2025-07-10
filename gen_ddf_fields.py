import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.example import standard_masks
from rubin_scheduler.scheduler.surveys import FieldSurvey
from rubin_scheduler.utils import ddf_locations

__all__ = [
    "gen_ddf_field_surveys",
]


def gen_ddf_field_surveys(
    nside=32,
    camera_ddf_rot_limit=75.0,
    wind_speed_maximum=20.0,
    moon_limit=30.0,
):
    """Generate a survey object for observing the Roman field(s)
    in an on season"""
    ddf_fields = ddf_locations()
    ddf_surveys = []
    for ddf_name in ddf_fields:
        if ddf_name == "EDFS_b":
            continue

        field_info = ddf_fields[ddf_name]

        RA = field_info[0]
        dec = field_info[1]

        scheduler_note = f"DD: {ddf_name}"
        target_name = ddf_name

        # Add some feasibility basis functions.
        basis_functions = standard_masks(
            nside,
            moon_distance=moon_limit,
            wind_speed_maximum=wind_speed_maximum,
            shadow_minutes=50,
        )
        basis_functions.append(bf.NotTwilightBasisFunction())
        # Force it to delay about a day (like VisitGap)
        basis_functions.append(
            bf.ForceDelayBasisFunction(
                days_delay=30.0 / 24, scheduler_note=scheduler_note
            )
        )
        # Add slewtime to choose nearest field
        basis_functions.append(bf.SlewtimeBasisFunction(bandname=None, nside=nside))

        # Add a dither detailer, so it dithers between each set
        # of exposures I guess?
        details = []
        details.append(
            detailers.DitherDetailer(max_dither=0.2, seed=42, per_night=False)
        )
        details.append(
            detailers.CameraRotDetailer(
                min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit
            )
        )

        survey = FieldSurvey(
            basis_functions,
            RA=RA,
            dec=dec,
            sequence="ugrizy",
            nexps=1,
            nvisits={"u": 10, "g": 10, "r": 20, "i": 20, "z": 25, "y": 20},
            exptimes={"u": 38, "g": 30, "r": 30, "i": 30, "z": 30, "y": 30},
            survey_name=scheduler_note,
            target_name=target_name,
            observation_reason="DDF",
            detailers=details,
        )
        ddf_surveys.append(survey)

    return ddf_surveys
