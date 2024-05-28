class ThubanWarning(Warning):
    """
    Warning class for Thuban
    """


class ConvergenceWarning(ThubanWarning):
    """
    A warning class to indicate when solving pointing or distortion does not converge
    """


class RepeatedStarWarning(ThubanWarning):
    """
    A warning class to indicate when the pointing or distortion requires reusing a star because too few provided
    """
