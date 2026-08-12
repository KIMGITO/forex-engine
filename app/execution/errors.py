"""Domain errors for the execution layer (Step 16)."""


class ExecutionError(Exception):
    """Base class for execution-layer domain errors."""


class InvalidOrderError(ExecutionError):
    """An order request is structurally invalid (bad symbol/quantity/stop)."""


class RiskGateViolationError(ExecutionError):
    """An order was submitted to a broker without a passed risk decision.

    The execution layer must never bypass the risk engine; a BrokerAdapter
    receiving such an order raises this error.
    """


class OrderNotFoundError(ExecutionError):
    """The requested order does not exist."""


class PositionNotFoundError(ExecutionError):
    """The requested position does not exist."""