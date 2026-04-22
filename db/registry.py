from typing import Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")
E = TypeVar("E")


class Registry(Generic[T, E]):
    """Generic registry for mapping values to classes"""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[E, Type[T]] = {}

    def _registry_lookup(self, value: E) -> Type[T]:
        if value not in self._registry:
            raise ValueError(
                f"{self.name}: Unknown type '{value}'. "
                f"Available: {self.get_available()}"
            )
        return self._registry[value]

    def register(self, value: E) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class"""

        def decorator(cls: Type[T]) -> Type[T]:
            if value in self._registry:
                raise ValueError(f"{self.name}: {value} is already registered")
            self._registry[value] = cls
            return cls

        return decorator

    def create(self, value: E, *args, **kwargs) -> T:
        """Factory method to create instances"""
        cls = self._registry_lookup(value)
        return cls(*args, **kwargs)

    def get_class(self, value: E) -> Type[T]:
        """Get the class without instantiating"""
        cls = self._registry_lookup(value)
        return cls

    def get_available(self) -> List[E]:
        """Get all registered values"""
        return list(self._registry.keys())

    def __contains__(self, value: E) -> bool:
        """Check if a value is registered"""
        return value in self._registry


class TypeHandlerRegistry(Registry[T, Type]):
    """
    Registry for mapping types to handlers.

    Resolves subclass relationships automatically.
    """

    def _registry_lookup(self, value: Type) -> Type[T]:
        if value in self._registry:
            return self._registry[value]
        for k, v in self._registry.items():
            if issubclass(value, k):
                return v
        raise ValueError(
            f"{self.name}: Unknown type '{value}'. Available: {self.get_available()}"
        )
