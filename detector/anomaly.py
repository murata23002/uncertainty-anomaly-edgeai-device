from typing import List

import cv2
import numpy as np

from images.image_util import Preprocessor

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

from config_manager.config import TFliteConfig
from logger.custom_logger import custom_logger, log_debug_method_execution


class AnomalyInferenceResult:
    def __init__(self):
        self.layer_outputs = []

    def set_results(self, all_layer_outputs):
        self.layer_outputs = all_layer_outputs

    def has_results(self):
        return self.layer_outputs is not None and len(self.layer_outputs) > 0

    def get_results(self):
        if self.has_results():
            return {"layer_outputs": self.layer_outputs}
        else:
            return None


class TFLiteClassifcation:
    _config = TFliteConfig(section_name="anomaly")

    def __init__(self):
        try:
            self.interpreter = tflite.Interpreter(
                model_path=self._config.model, num_threads=self._config.num_threads
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.input_height = self.input_details[0]["shape"][1]
            self.input_width = self.input_details[0]["shape"][2]
            self.input_type = self.input_details[0]["dtype"]

            custom_logger.info(
                f"EfficientNet model loaded successfully / "
                f"model name {self._config.model} / "
                f"height {self.input_height} / "
                f"input_width {self.input_width} / "
                f"input_type {self.input_type}"
            )

        except Exception as e:
            custom_logger.exception("Error loading EfficientNet model")
            raise e

    @log_debug_method_execution()
    def run_inference(self, input_data):
        try:
            all_layer_outputs = [[] for _ in range(9)]

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            for i, output_detail in enumerate(self.output_details):
                output_data = self.interpreter.get_tensor(output_detail["index"])
                all_layer_outputs[i].append(output_data.squeeze())

            return all_layer_outputs

        except Exception:
            custom_logger.exception("An error occurred during EfficientNet inference")
            return None


class Anomaly:
    def __init__(self):
        self.model = TFLiteClassifcation()
        self.preprocessor = Preprocessor(
            self.model.input_width, self.model.input_height, self.model.input_type
        )
        custom_logger.info("EfficientNet detector initialization is complete")

    def detect(self, frame):
        result = AnomalyInferenceResult()
        input_data = self.preprocessor.process(frame, is_normalizing=False)

        all_layer_outputs = self.model.run_inference(input_data)
        result.set_results(all_layer_outputs)

        return result.get_results()
