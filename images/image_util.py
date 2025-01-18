import cv2
import numpy as np

from logger.custom_logger import custom_logger


class Preprocessor:
    def __init__(self, input_width, input_height, input_type):
        self.input_width = input_width
        self.input_height = input_height
        self.input_type = input_type

    def process(self, frame, is_normalizing=True):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.resize(rgb_frame, (self.input_width, self.input_height))
            if self.input_type == np.float32:
                if is_normalizing:
                    rgb_frame = rgb_frame.astype(np.float32) / 255.0
                else:
                    rgb_frame = rgb_frame.astype(np.float32)
            else:
                rgb_frame = rgb_frame.astype(self.input_type)

            rgb_frame = np.expand_dims(rgb_frame, axis=0)
            return rgb_frame
        except Exception:
            custom_logger.exception("Preprocessing failed")
            return None


class ImgaesOpretion:
    pass
