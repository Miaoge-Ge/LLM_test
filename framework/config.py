import os
import yaml
from typing import Dict, Any, Optional

class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: str = "config.yaml"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        
        with open(self._config_path, "r", encoding="utf-8") as f:
            try:
                self._config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config file: {e}")

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get_selected_model_config(self) -> Dict[str, Any]:
        """
        Retrieves the configuration for the currently selected model.
        Resolves provider information automatically.
        """
        selected_model_key = self._config.get("selected_model")
        if not selected_model_key:
            raise ValueError("No 'selected_model' defined in config.yaml")
        
        models = self._config.get("models", {})
        model_config = models.get(selected_model_key)
        
        if not model_config:
            raise ValueError(f"Model config for '{selected_model_key}' not found in 'models' section")
            
        # Resolve provider
        provider_key = model_config.get("provider")
        if not provider_key:
            raise ValueError(f"No 'provider' specified for model '{selected_model_key}'")
            
        providers = self._config.get("providers", {})
        provider_config = providers.get(provider_key)
        
        if not provider_config:
            raise ValueError(f"Provider '{provider_key}' not found in 'providers' section")
            
        # Merge provider config with model config (model config takes precedence if keys conflict, though unlikely here)
        final_config = provider_config.copy()
        final_config.update(model_config)
        
        return final_config

    def get_dataset_path(self, task_name: str) -> str:
        datasets = self._config.get("datasets", {})
        path = datasets.get(task_name)
        if not path:
             # Fallback or error
             return ""
        return path

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def reload(self):
        """Reload configuration from disk"""
        self._load_config()
