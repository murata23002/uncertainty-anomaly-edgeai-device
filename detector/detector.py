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


class DetectInferenceResult:
    def __init__(self):

        self.num = None
        self.class_ids = []
        self.class_label = []
        self.boxes = []
        self.scores = []

    def _get_class_labels(self, label_map, class_ids: List[int]) -> List[str]:
        return [label_map.get(class_id, f"Class {class_id}") for class_id in class_ids]

    def set_results(self, label_map, num, class_ids, boxes, scores):
        self.boxes = boxes
        self.class_ids = class_ids
        self.class_label = self._get_class_labels(label_map, class_ids)
        self.scores = scores
        self.num = num

    def has_results(self):
        return self.num is not None

    def get_results(self):
        if self.has_results():
            return {
                "num": self.num,
                "ids": self.class_ids,
                "labels": self.class_label,
                "boxes": self.boxes,
                "scores": self.scores,
            }
        else:
            return None


class TFLiteDetect:
    _config = TFliteConfig(section_name="detector")

    def __init__(self):
        try:
            if self._config.enable_edgetpu:
                # TODO maybe not work
                delegates = [tflite.load_delegate("libedgetpu.so.1")]
                self.interpreter = tflite.Interpreter(
                    model_path=self._config.model,
                    experimental_delegates=delegates,
                )
            else:
                self.interpreter = tflite.Interpreter(
                    model_path=self._config.model,
                    num_threads=self._config.num_threads,
                )

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_height = self.input_details[0]["shape"][1]
            self.input_width = self.input_details[0]["shape"][2]
            self.input_type = self.input_details[0]["dtype"]

            custom_logger.info(
                f"model loaded successfully / "
                f"model name {self._config.model} / "
                f"height {self.input_height} / "
                f"input_width {self.input_width} / "
                f"input_type {self.input_type}"
            )

        except Exception as e:
            custom_logger.exception("model loaded error")
            raise e

    @log_debug_method_execution()
    def run_inference(self, input_data, img_width, img_height):
        try:
            boxes = []
            class_ids = []
            scores = []

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            boxes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
            num = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
            class_ids = self.interpreter.get_tensor(self.output_details[3]["index"])[0]

            boxes[:, 0] = (boxes[:, 0] * img_width).astype(int)
            boxes[:, 1] = (boxes[:, 1] * img_height).astype(int)
            boxes[:, 2] = (boxes[:, 2] * img_width).astype(int)
            boxes[:, 3] = (boxes[:, 3] * img_height).astype(int)

            return (
                num.astype(int),
                class_ids.astype(int),
                boxes.astype(int),
                scores,
            )

        except Exception:
            custom_logger.exception("an error occurred during inference")
            return None, None, None


class Detector:
    def __init__(self, label_map):
        self.model = TFLiteDetect()
        self.preprocessor = Preprocessor(
            self.model.input_width, self.model.input_height, self.model.input_type
        )
        self.label_map = label_map

        custom_logger.info("detector initialization is complete")

    def detect(self, frame):
        result = DetectInferenceResult()
        img_width, img_height, _ = frame.shape
        input_data = self.preprocessor.process(frame)
        if input_data is None:
            custom_logger.warning("preprocessing failed")
            return result

        (
            num,
            class_ids,
            boxes,
            scores,
        ) = self.model.run_inference(input_data, img_width, img_height)

        result.set_results(self.label_map, num, class_ids, boxes, scores)

        return result
