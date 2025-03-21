import numpy as np
from rubin_scheduler.scheduler.utils import BasePixelEvolution


class PerFilterStep(BasePixelEvolution):
    """Make a custom step function per filter
    """
    def __init__(self, survey_length=80, nfilters=6, u_loaded=None, y_loaded=None):
        self.nfilters = nfilters
        self.survey_length = survey_length

        if u_loaded is None:
            self.u_loaded = np.arange(survey_length)
        else:
            self.u_loaded = u_loaded

        if y_loaded is None:
            self.y_loaded = np.arange(survey_length)
        else:
            self.y_loaded = y_loaded

        self.u_slope = 1./self.u_loaded.size
        self.y_slope = 1./self.y_loaded.size

    def __call__(self, t_elapsed, phase):

        # filters all the time evolve linearly increase between
        # 0 and 1 for length of survey.
        frac_done = t_elapsed/self.survey_length

        # broadcast out to n_filters
        result = np.tile(phase*0 + frac_done, (self.nfilters, 1))

        # u_band
        n_u = np.where(self.u_loaded <= t_elapsed)[0].size
        u_result = result[0, :]*0 + n_u*self.u_slope
        result[0, :] = u_result

        # y band
        n_y = np.where(self.y_loaded <= t_elapsed)[0].size
        y_result = result[-1, :]*0 + n_y*self.y_slope
        result[-1, :] = y_result

        return result

