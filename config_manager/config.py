import json
import os
from typing import Any, Dict, Union


class BaseConfig:
    _shared_config: Union[None, Dict[str, Any]] = None
    _config_files_directory = "config"

    def __init__(self, file_name: str) -> None:
        self.program_directory = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(
            os.path.dirname(self.program_directory),
            self._config_files_directory,
            file_name,
        )
        self.config: Dict[str, Any] = self._load_json(self.config_path)

    def get_config(self, keys: str, default=None):
        if not isinstance(self.config, dict):
            return default
        keys_list = keys.split(".")
        data = self.config
        for key in keys_list:
            if key in data:
                data = data[key]
            else:
                return default
        return data

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{file_path}': {e}") from e


class TFliteConfig(BaseConfig):
    def __init__(self, section_name) -> None:
        super().__init__("tflite_config.json")
        self.section_name = section_name

    @property
    def model(self) -> str:
        return self.get_config(f"{self.section_name}.model", "model.tflite")

    @property
    def enable_edgetpu(self) -> bool:
        return self.get_config(f"{self.section_name}.enable_edgetpu", False)

    @property
    def num_threads(self) -> int:
        return self.get_config(f"{self.section_name}.num_threads", 4)

    @property
    def score_threshold(self) -> float:
        return self.get_config(f"{self.section_name}.score_threshold", 0.5)


class LogConfigs(BaseConfig):
    def __init__(self) -> None:
        super().__init__("log_configs.json")

    @property
    def level(self) -> str:
        return self.get_config("log.level", "INFO")

    @property
    def file(self) -> str:
        return self.get_config("log.file", "logs.txt")

    @property
    def encoding(self) -> str:
        return self.get_config("log.encoding", "UTF-8")


class ApiConfigs(BaseConfig):
    def __init__(self, section_name: str) -> None:
        super().__init__("api_config.json")
        self.section_name = section_name

    @property
    def url(self) -> str:
        return self.get_config(f"{self.section_name}.url", "")

    @property
    def headers(self) -> Dict[str, str]:
        return self.get_config(
            f"{self.section_name}.headers", {"Content-Type": "application/json"}
        )

    @property
    def auth(self) -> Union[None, tuple]:
        username = self.get_config(f"{self.section_name}.username", "")
        password = self.get_config(f"{self.section_name}.password", "")
        if username and password:
            return (username, password)
        else:
            return None

    @property
    def timeout(self) -> int:
        return self.get_config(f"{self.section_name}.timeout", 10)


class LabelConfigs(BaseConfig):
    def __init__(self, config_file: str = "label_map.json") -> None:
        super().__init__(config_file)
        self.label_map = self._load_label_map()

    def _load_label_map(self) -> Dict[int, str]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}
            return label_map
        except Exception:
            raise
