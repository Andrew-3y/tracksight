from pathlib import Path

import cv2
import numpy as np


class VehicleReIdentifier:
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Could not find vehicle re-identification model: {model_path}"
            )
        self.network = cv2.dnn.readNetFromONNX(str(model_path))

    def get_embedding(self, frame, box):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = box
        left = max(0, min(frame_width, round(x1)))
        top = max(0, min(frame_height, round(y1)))
        right = max(0, min(frame_width, round(x2)))
        bottom = max(0, min(frame_height, round(y2)))
        vehicle_crop = frame[top:bottom, left:right]
        if vehicle_crop.size == 0:
            return None
        input_blob = cv2.dnn.blobFromImage(
            vehicle_crop,
            size=(208, 208),
            swapRB=True,
        )
        self.network.setInput(input_blob)
        embedding = self.network.forward().flatten().astype(np.float32)
        embedding_length = np.linalg.norm(embedding)
        if embedding_length == 0:
            return None
        return embedding / embedding_length


def calculate_appearance_distance(embedding_a, embedding_b):
    if embedding_a is None or embedding_b is None:
        return None
    cosine_similarity = float(np.dot(embedding_a, embedding_b))
    return 1 - cosine_similarity
