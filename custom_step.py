import numpy as np
from rubin_scheduler.scheduler.utils import BasePixelEvolution


class PerFilterStep(BasePixelEvolution):
    """Make a custom step function per filter"""

    def __init__(
        self,
        survey_length=80,
        bands=None,
        loaded_dict=None,
    ):

        if bands is None:
            bands = ["u", "g", "r", "i", "z", "y"]
        if loaded_dict is None:
            loaded_dict = {}

        self.bands = bands
        self.survey_length = survey_length

        self.bands2indx = {}
        for i, bandname in enumerate(bands):
            self.bands2indx[bandname] = i

        self.slopes = {}
        self.loaded_dict = {}

        for bandname in bands:
            if bandname in loaded_dict.keys():
                self.loaded_dict[bandname] = loaded_dict[bandname]
                self.slopes[bandname] = 1.0 / loaded_dict[bandname].size

    def __call__(self, t_elapsed, phase):
        """
        Parameters
        ----------
        t_elapsed : `float`
            Time elapsed in the survey (days).
        """

        # filters all the time evolve linearly increase between
        # 0 and 1 for length of survey.
        frac_done = t_elapsed / self.survey_length

        # broadcast out to n_filters
        result = np.tile(phase * 0 + frac_done, (len(self.bands), 1))

        for bandname in self.loaded_dict:
            days_completed = np.where(self.loaded_dict[bandname] <= t_elapsed)[0].size
            result[self.bands2indx[bandname], :] = (
                result[self.bands2indx[bandname], :] * 0
                + days_completed * self.slopes[bandname]
            )

        return result
