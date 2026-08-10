"""Domain exceptions for the research layer."""


class ResearchError(Exception):
    """Base exception for all research-layer errors."""


class DatasetConfigurationError(ResearchError):
    """Raised when research dataset configuration is invalid."""


class SplitConfigurationError(ResearchError):
    """Raised when split configuration is invalid."""


class LeakagePreventionError(ResearchError):
    """Raised when optimization/splitting would leak test data into training."""


class OptimizerError(ResearchError):
    """Raised when optimization fails."""