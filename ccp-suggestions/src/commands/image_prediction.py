import cv2
import json
from datetime import datetime as dt
from google.cloud import storage
from io import BytesIO
from ultralytics import YOLO

from src.static.constants import model_path, bucket_name, bucket_image_folder, recommendations_topic
from src.commands.common.pubsub import publish_message


class PredictionModel:
    def __init__(
        self,
        body
    ):
        self.blob_path = body['BlobPath']
        self.model_path = model_path
        self.model = self.model_initialization()

    def model_initialization(self):
        model = YOLO(self.model_path)
        return model

    def predict_keyframes(self, model, num_frames: int = 10):
        capture = cv2.VideoCapture(self.blob_path)
        if not capture.isOpened():
            raise IOError(f"Failed to open video: {self.blob_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames)

        current = 0
        processed = 0

        predictions = []
        while processed < num_frames and capture.isOpened():
            capture.set(cv2.CAP_PROP_POS_FRAMES, current)
            ret, frame = capture.read()
            if not ret:
                break

            result = model(frame, save=False)
            predictions.append(self.draw_predictions(result, frame))
            current += step
            processed += 1

        capture.release()
        return predictions

    def draw_predictions(self, result, image):
        boxes = result[0].boxes
        boxes_data = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{self.model.names[cls]} {conf:.2f}"

            boxes_data.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class_id": cls,
                    "label": label,
                }
            )

            # Draw rectangle with thinner line
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

            # Draw label with smaller font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            (label_width, label_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw filled rectangle for label background
            cv2.rectangle(
                image,
                (x1, y1 - label_height - 4),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1,
            )
            # Put text label
            cv2.putText(
                image, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), font_thickness
            )

        success, encoded_image = cv2.imencode(".jpg", image)
        if not success:
            RuntimeError("Failed to encode image")

        image_bytes = BytesIO(encoded_image.tobytes())
        return {
            "image": image_bytes,
            "boxes": len(boxes),
            "boxes_data": json.dumps(boxes_data)
        }

    def execute(self):
        predictions = self.predict_keyframes(self.model)
        top_5_predictions = sorted(predictions, key=lambda x: x["boxes"], reverse=True)[
            :5
        ]

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        now = dt.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

        message = []
        
        for index, image in enumerate(top_5_predictions):
            blob = bucket.blob(f"{bucket_image_folder}/frame_{index}_{now}.jpg")
            image["image"].seek(0)
            blob.upload_from_file(image["image"], content_type="image/jpeg")
            meta_blob = bucket.blob(f"{bucket_image_folder}/frame_metadata_{index}_{now}.json")
            meta_blob.upload_from_string(image["boxes_data"], content_type="application/json")

            message.append({
                "image": f"frame_{index}_{now}.jpg",
                "metadata": f"frame_metadata_{index}_{now}.json"
            })
        
        r = publish_message(recommendations_topic, message)
        return {"response": r, "status_code": 200}