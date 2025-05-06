import cv2
# from imread_from_url import imread_from_url
from yolov8 import YOLOv8

#Initialize yolov8
model_path = "models/best18.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.25, iou_thres=0.25)

#Read image
# img_url = "https://media.newyorker.com/photos/5e5ed01f39e0e500082b73b6/master/w_2560%2Cc_limit/Rosner-CoronavirusPanicShopping.jpg"
# img = imread_from_url(img_url)
img_list = ["carulla.jpeg", "Quieres somos_0.jpg", "maxresdefault.jpg"]


#Draw detections
for item in img_list:
    image = cv2.imread(item)
    boxes, scores, class_ids = yolov8_detector(image) #Detect Objects
    print(boxes)
    combined_img = yolov8_detector.draw_detections(image)
    cv2.imwrite(f"test/images/detected_{item}", combined_img)
