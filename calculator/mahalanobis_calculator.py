import os
import pickle

import numpy as np


class Mean_inv_Cov_Data:
    def __init__(self, class_id):
        self.class_id = class_id
        self.data = []

    def add(self, index, mean_feat, inv_cov_feat):
        self.data.append(
            {"layer_index": index, "mean_feat": mean_feat, "inv_cov_feat": inv_cov_feat}
        )

    def get_layer_data(self, index):
        return self.data[index]


class MeanInvCovDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_all_class_data(self):
        class_data_dict = {}
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith("_mean_inv_cov.pkl"):
                class_id = file_name.split("_mean_inv_cov.pkl")[0]
                class_data = self.load_data_by_class_id(class_id)
                class_data_dict[class_id] = class_data
        return class_data_dict

    def load_data_by_class_id(self, class_id):
        file_name = f"{class_id}_mean_inv_cov.pkl"
        file_path = os.path.join(self.data_dir, file_name)
        class_data = Mean_inv_Cov_Data(class_id)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                layer_data_list = pickle.load(f)
                for index in layer_data_list:
                    mean_feat = layer_data_list[index]["mean_feat"]
                    inv_cov_feat = layer_data_list[index]["inv_cov_feat"]
                    class_data.add(index, mean_feat, inv_cov_feat)
            return class_data
        else:
            raise FileNotFoundError(
                f"Data with class ID {class_id} not found in {self.data_dir}"
            )


class VectorMetrics:
    def __init__(self, class_data_dict):
        self.class_data_dict = class_data_dict

    def _validate_vector(self, u, dtype=None):
        u = np.asarray(u, dtype=dtype, order="c")
        if u.ndim == 1:
            return u
        raise ValueError("Input vector should be 1-D.")

    def mahalanobis(self, u, v, VI):
        u = self._validate_vector(u)
        v = self._validate_vector(v)
        VI = np.atleast_2d(VI)
        delta = u - v
        m = np.dot(np.dot(delta, VI), delta)
        return np.sqrt(m)

    def distances(self, class_id, sample_feats):
        if class_id in self.class_data_dict:
            class_data = self.class_data_dict[class_id]
            total_distance = 0
            for index, sample_feat in enumerate(sample_feats):
                layer_data = class_data.get_layer_data(index)
                mean_feat = layer_data["mean_feat"]
                inv_cov_feat = layer_data["inv_cov_feat"]

                dist = self.mahalanobis(
                    sample_feat[0],
                    mean_feat,
                    inv_cov_feat,
                )
                total_distance += dist
            return total_distance
        else:
            raise KeyError(f"Class ID {class_id} not found in class_data_dict")

    def angle_difference_sum(self, class_id, sample_feats):
        if class_id in self.class_data_dict:
            class_data = self.class_data_dict[class_id]
            total_angle_diff = 0
            for index, sample_feat in enumerate(sample_feats):
                layer_data = class_data.get_layer_data(index)
                mean_feat = layer_data["mean_feat"]

                sample_vec = np.asarray(sample_feat[0])
                mean_vec = np.asarray(mean_feat)

                dot_product = np.dot(sample_vec, mean_vec)
                norm_sample = np.linalg.norm(sample_vec)
                norm_mean = np.linalg.norm(mean_vec)
                cosine_similarity = dot_product / (norm_sample * norm_mean)

                angle_diff = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))
                angle_diff_degrees = np.degrees(angle_diff)

                total_angle_diff += angle_diff_degrees
            return total_angle_diff
        else:
            raise KeyError(f"Class ID {class_id} not found in class_data_dict")
