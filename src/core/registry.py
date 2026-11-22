"""Generic component registry for modular backend lookup."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry for factories or component classes."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a component factory under a canonical name."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._items[name] = factory
            return factory

        return decorator

    def add(self, name: str, factory: Callable[..., T]) -> None:
        """Imperatively register a component factory."""
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., T]:
        """Retrieve a registered factory or raise KeyError with helpful options."""
        if name not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._items[name]

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate a registered component."""
        return self.get(name)(*args, **kwargs)

    def list_names(self) -> list[str]:
        """Return sorted list of registered component names."""
        return sorted(self._items.keys())
