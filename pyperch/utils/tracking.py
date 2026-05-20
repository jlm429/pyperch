from dataclasses import dataclass


@dataclass
class OptimizerSnapshot:
    """Serializable snapshot of optimizer counters."""

    function_evals: int
    proposed_steps: int
    accepted_steps: int
    rejected_steps: int
    best_loss: float | None

    @classmethod
    def from_optimizer(cls, optimizer):
        """Create a snapshot from an optimizer with standard counters."""
        return cls(
            function_evals=optimizer.function_evals,
            proposed_steps=optimizer.proposed_steps,
            accepted_steps=optimizer.accepted_steps,
            rejected_steps=optimizer.rejected_steps,
            best_loss=optimizer.best_loss,
        )
