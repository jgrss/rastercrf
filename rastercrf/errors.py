class FitError(AttributeError):
    """Raised when the classifier has not been fitted and, therefore, cannot be used for predictions"""


class TimeError(Exception):

    """Raised when the number of bands do not match across all states"""

    def __init__(self, xi, x_state_shape, nattrs):
        self.message = f'State {xi+1} does not have a matching band count to the fitted features. There are {x_state_shape} bands but the model was fit on {nattrs} features.'
        super().__init__(self.message)

    def __str__(self):
        return self.message
