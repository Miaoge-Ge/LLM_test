from typing import Dict, Type, Any, List

class TaskRegistry:
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: Type[Any]):
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        return cls._registry.get(name)

    @classmethod
    def list_tasks(cls) -> List[str]:
        return list(cls._registry.keys())
