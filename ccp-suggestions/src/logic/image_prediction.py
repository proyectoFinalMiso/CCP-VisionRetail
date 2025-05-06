import cv2
from ultralytics import YOLO

from src.static.constants import model_path

class PredictionModel:

    def __init__(self, video_blob_path):
        self.blob_path = video_blob_path
        self.model_path = model_path

    def model_initialization(self):
        model = YOLO(self.model_path)
        return model
    
    def predict_keyframes(self, model, num_frames: int = 20):
        capture = cv2.VideoCapture(self.blob_path)
        if not capture.isOpened():
            raise IOError(f"Failed to open video: {self.blob_path}")
        
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames)

        current = 0
        processed = 0

        while processed < num_frames and capture.isOpened():
            capture.set(cv2.CAP_PROP_POS_FRAMES, current)
            ret, frame = capture.read()
            if not ret:
                break

            results = model(frame, save=False)
            self.save_prediction(results, frame, f"frame_{processed}.jpg")

            current += step
            processed += 1

        capture.release()

    def save_prediction(self, results, image, name: str):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"

                # Draw rectangle with thinner line
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

                # Draw label with smaller font
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

                # Draw filled rectangle for label background
                cv2.rectangle(image, (x1, y1 - label_height - 4), (x1 + label_width, y1), (0, 255, 0), -1)
                # Put text label
                cv2.putText(image, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

        cv2.imwrite(f"runs/prediction_{name}", image)