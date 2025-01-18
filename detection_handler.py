import json
import os
from datetime import datetime

import cv2
import numpy as np
import requests

from calculator.mahalanobis_calculator import MeanInvCovDataLoader, VectorMetrics
from config_manager .config import ApiConfigs
from detector.anomaly import Anomaly
from logger.custom_logger import custom_logger


class Drawer:
    def __init__(
        self, enable_drawing: bool = True, detect_score_threshold: float = 0.2
    ):
        self.enable_drawing = enable_drawing
        self.detect_score_threshold = detect_score_threshold

    def draw(self, frame, cls_id, cls_label, box, score, distances, angle_diff):
        if not self.enable_drawing:
            return
        try:
            if score <= 0.1:
                return
            x1, y1 = box[0], box[1]
            x2, y2 = box[2], box[3]

            text = f"{cls_label} score: {score:.3f} Dist: {distances:.3f} Angle_diff: {angle_diff:.3f}"

            font_scale = 0.4  # 文字サイズ
            font_thickness = 1  # 文字の太さ
            text_color = (255, 255, 255)  # 白色の文字
            bg_color = (0, 0, 0)  # 黒色の背景

            # テキストサイズを取得して背景用の矩形を描画
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # テキストの表示位置を調整
            text_x = x1  # バウンディングボックスの左上に揃える
            text_y = y1 - 5  # バウンディングボックスの少し上に配置

            # テキストの背景を描画
            cv2.rectangle(
                frame,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                bg_color,
                cv2.FILLED,
            )

            # テキストを描画
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

            # 矩形を描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            custom_logger.debug("result drawing has completed successfully")
        except Exception:
            custom_logger.exception("error occurred while drawing the result")


class Sender:
    def __init__(self, server_url, headers=None, auth=None, timeout=10):
        self.server_url = server_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.auth = auth
        self.timeout = timeout
        custom_logger.info("sender initialisation is complete")

    def send(self, data):
        try:
            response = requests.post(
                self.server_url,
                json=data,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,
            )
            response.raise_for_status()
            custom_logger.info(
                f"data transmission was successful status code: {response.status_code}"
            )
            return True
        except Exception:
            custom_logger.exception("error occurred while sending data")
            return False


class ObjectDetectHandler:
    def __init__(
        self,
        api_config: ApiConfigs,
        detect_score_threshold,
        enable_drawing: bool,
        mean_inv_cov_path,
    ):
        self.drawer = Drawer(enable_drawing, detect_score_threshold)
        self.sender = Sender(
            server_url=api_config.url,
            headers=api_config.headers,
            auth=api_config.auth,
            timeout=api_config.timeout,
        )
        self.sender = Sender(
            server_url=api_config.url,
            headers=api_config.headers,
            auth=api_config.auth,
            timeout=api_config.timeout,
        )
        self.anomaly = Anomaly()
        data_loader = MeanInvCovDataLoader(mean_inv_cov_path)
        mean_inv_cov_dicts = data_loader.load_all_class_data()
        self.metric = VectorMetrics(mean_inv_cov_dicts)

        custom_logger.debug("DetectionHandler initialization is complete")

    def save_frame(self, frame, output_directory, timestamp):
        file_path = os.path.join(output_directory, f"frame_{timestamp}.png")
        cv2.imwrite(file_path, frame)
        custom_logger.info(f"Frame saved: {file_path}")

    def _clip_image_by_box(self, frame, box, img_width, img_height):
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        w, h = x2 - x1, y2 - y1

        if h > 0 and w > 0:
            x1 = np.clip(x1, 0, img_width)
            y1 = np.clip(y1, 0, img_height)
            x2 = np.clip(x2, 0, img_width)
            y2 = np.clip(y2, 0, img_height)
            return frame[y1:y2, x1:x2]
        else:
            return None

    def _get_distances(self, cls_label, sample_feats):
        distances = 0
        if sample_feats is not None:
            distances = self.metric.distances(cls_label, sample_feats["layer_outputs"])
            custom_logger.debug(f"distances is {distances}")

        return distances

    def _get_angle_diff(self, cls_label, sample_feats):
        diff = 0
        if sample_feats is not None:
            diff = self.metric.angle_difference_sum(
                cls_label, sample_feats["layer_outputs"]
            )
            custom_logger.debug(f"angle diff is {diff}")

        return diff

    def save_results_to_json(
        self,
        frame,
        output_directory,
        output_frame,
        num,
        class_ids,
        class_labels,
        boxes,
        scores,
        timestamp,
    ):
        file_path = os.path.join(output_directory, f"detect_{timestamp}.json")

        json_result = []
        height, width = frame.shape[:2]

        for i in range(num):
            score = scores[i]
            cls_id = class_ids[i]
            cls_label = class_labels[i]
            x1, y1 = boxes[i][1], boxes[i][0]
            x2, y2 = boxes[i][3], boxes[i][2]

            distances = 0
            angle_diff = 0
            if score >= 0.1:
                cropped_img = self._clip_image_by_box(
                    frame, (x1, y1, x2, y2), width, height
                )
                if cropped_img is not None:
                    sample_feats = self.anomaly.detect(cropped_img)

                    distances = self._get_distances(cls_label, sample_feats)
                    angle_diff = self._get_angle_diff(cls_label, sample_feats)

            result = {
                "class_id": int(cls_id),
                "class_label": cls_label,
                "score": float(score),
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "anomaly_distances": distances,
                "angle_diff": angle_diff,
            }
            self.drawer.draw(
                frame, cls_id, cls_label, (x1, y1, x2, y2), score, distances, angle_diff
            )
            json_result.append(result)

        self.save_frame(frame, output_frame, timestamp)

        with open(file_path, "w") as f:
            json.dump(json_result, f, indent=2)

    def create_output_directory(self, directory="output", name="dist"):
        output_directory = os.path.join(directory, name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory

    # TODO 処理を詰め込みすぎなので分離する。クラス設計の見直し
    def process_results(self, frame, num, class_ids, class_labels, boxes, scores):
        output_frame = self.create_output_directory(name="frames")
        output_detect = self.create_output_directory(name="detect")
        output_images = self.create_output_directory(name="images")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.save_frame(frame, output_images, timestamp)
        self.save_results_to_json(
            frame,
            output_detect,
            output_frame,
            num,
            class_ids,
            class_labels,
            boxes,
            scores,
            timestamp,
        )

        # self.sender.send(parsed_results)
