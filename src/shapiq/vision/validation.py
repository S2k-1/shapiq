"""Shared validation helpers for vision model compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shapiq.typing import Model


class ModelCompatible:
    """Trait for strategies that validate a compatible model protocol.

    Subclasses declare the model protocol they accept via
    ``compatible_model_protocol`` and inherit a shared ``validate_model``
    implementation that raises a ``TypeError`` for incompatible models.
    """

    compatible_model_protocol: type | tuple[type, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclasses declare a compatible model protocol."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "compatible_model_protocol"):
            msg = f"{cls.__name__} must define 'compatible_model_protocol'."
            raise TypeError(msg)

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol.

        Args:
            model: Object to validate against ``compatible_model_protocol``.

        Raises:
            TypeError: If ``model`` is not compatible with the declared
                protocol.
        """
        protocol = cls.compatible_model_protocol

        if not isinstance(model, protocol):
            if isinstance(protocol, tuple):
                expected = ", ".join(proto.__name__ for proto in protocol)
            else:
                expected = protocol.__name__

            msg = (
                f"{cls.__name__} requires a model compatible with {expected}, "
                f"got {type(model).__name__}."
            )
            raise TypeError(msg)
