import json
from typing import Any, Dict, Optional

class ConfigManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get_config(self, key: str, default: Optional[Any] = None) -> Any:
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        self.config[key] = value
        self._save_config()

    def _save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

config_manager = ConfigManager()