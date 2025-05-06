import cv2

# from imread_from_url import imread_from_url
from ultralytics import YOLO

#Initialize yolov8
model_path = "models/bestv8.pt"
model = YOLO(model_path, task="detect")

img_list = ["carulla.jpeg", "Quieres somos_0.jpg", "maxresdefault.jpg"]
image = cv2.imread(img_list[2])

results = model(f"{img_list[2]}", save=False)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

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

cv2.imwrite(f"runs/prediction_{img_list[2]}", image)