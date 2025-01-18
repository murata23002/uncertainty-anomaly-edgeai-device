import argparse
import os

import cv2

from config_manager.config import ApiConfigs, LabelConfigs, TFliteConfig
from detection_handler import ObjectDetectHandler
from detector.detector import Detector
from logger.custom_logger import custom_logger
from sensor.vision import Camera


def main(show_frame=False):
    custom_logger.info("Initialization starts")

    width = 640
    height = 480
    mean_inv_cov_path = f"{os.path.dirname(os.path.abspath(__file__))}/mean_inv_cov"

    detect_score_threshold = TFliteConfig(section_name="detector").score_threshold

    detector = Detector(label_map=LabelConfigs().label_map)
    detect_handler = ObjectDetectHandler(
        api_config=ApiConfigs(section_name="LINE"),
        enable_drawing=True,
        detect_score_threshold=detect_score_threshold,
        mean_inv_cov_path=mean_inv_cov_path,
    )

    try:
        with Camera(width=width, height=height) as cam:
            for frame in cam.iterate_frames():
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    custom_logger.info("Exit key pressed, closing camera.")
                    break
                results = detector.detect(frame).get_results()

                if results is not None:
                    detect_handler.process_results(
                        frame=frame,
                        num=results["num"],
                        class_ids=results["ids"],
                        class_labels=results["labels"],
                        boxes=results["boxes"],
                        scores=results["scores"],
                    )
                    if show_frame:
                        cv2.imshow("Image Window", frame)

    except Exception:
        custom_logger.exception("実行中にエラーが発生しました")
    finally:
        custom_logger.info("アプリケーションを終了します。")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object Detection with optional frame display."
    )
    parser.add_argument(
        "--show-frame",
        action="store_true",
        help="Display the frame in a window if this flag is set.",
    )
    args = parser.parse_args()

    main(show_frame=args.show_frame)
