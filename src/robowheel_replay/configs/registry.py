from typing import Type, Dict, List
from .schema import DatasetToSceneConfig

_DATASET_REGISTRY: Dict[str, Type[DatasetToSceneConfig]] = {}

def register_dataset_config(dataset_id: str):
    """
    Decorator to register a config class to the global registry.
    """
    def decorator(cls):
        _DATASET_REGISTRY[dataset_id] = cls
        return cls
    return decorator

def get_config_class(dataset_id: str) -> Type[DatasetToSceneConfig]:
    """
    Retrieve class from registry by ID.
    """
    if dataset_id not in _DATASET_REGISTRY:
        available_keys = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset ID '{dataset_id}' not found. Registered: {available_keys}")
    
    return _DATASET_REGISTRY[dataset_id]