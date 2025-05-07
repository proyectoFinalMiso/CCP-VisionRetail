import cv2
import json
import numpy as np
from google.cloud import storage
from src.static.constants import bucket_name, bucket_image_folder

class GenerateRecommendations:

    def download_filtered_images(self):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        images = []
        for i in range(5):
            img_name = f"frame_{i}.jpg"
            meta_name = f"frame_metadata_{i}.json"
            
            img_blob = bucket.blob(f"{bucket_image_folder}/{img_name}")
            meta_blob = bucket.blob(f"{bucket_image_folder}/{meta_name}")

            image_bytes = img_blob.download_as_bytes()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            json_bytes = meta_blob.download_as_bytes()
            metadata = json.loads(json_bytes.decode('utf-8'))

            images.append({
                'image': image,
                'metadata': metadata
            })

        return images