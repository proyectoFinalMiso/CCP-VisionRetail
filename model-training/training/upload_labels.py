import pandas as pd
import json
from collections import defaultdict

# === CONFIGURATION ===
CSV_PATH = "datasets/SKU-110K/annotations/annotations_train.csv"  # your file
GCS_BUCKET_URL = "https://storage.googleapis.com/ccp-training-data/images/"
LABEL_NAME = "object"  # all labels are "object"
OUTPUT_JSON = "label_studio_tasks.json"

# === READ CSV ===
columns = ["filename", "x1", "y1", "x2", "y2", "label", "img_width", "img_height"]
df = pd.read_csv(CSV_PATH, names=columns)

# === GROUP BY IMAGE ===
tasks = defaultdict(list)

for _, row in df.iterrows():
    img_url = GCS_BUCKET_URL + row["filename"]
    w = row["img_width"]
    h = row["img_height"]

    # Calculate normalized coordinates (percentages)
    x = (row["x1"] / w) * 100
    y = (row["y1"] / h) * 100
    width = ((row["x2"] - row["x1"]) / w) * 100
    height = ((row["y2"] - row["y1"]) / h) * 100

    annotation = {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rectanglelabels": [LABEL_NAME]
        }
    }

    tasks[img_url].append(annotation)

# === FORMAT FOR LABEL STUDIO ===
final_tasks = []
for img_url, results in tasks.items():
    task = {
        "data": { "image": img_url },
        "annotations": [ { "result": results } ]
    }
    final_tasks.append(task)

# === EXPORT JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(final_tasks, f, indent=2)

print(f"✅ Exported {len(final_tasks)} tasks to {OUTPUT_JSON}")
