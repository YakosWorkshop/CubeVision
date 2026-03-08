"""
Rubik's Cube Sticker Detection — Training Script
Run this once to download the dataset and train the model.
Trained weights will be saved to:
  runs/detect/rubiks_cube_detector/weights/best.pt
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# --- 1. Load API Key from .env ---
load_dotenv()

# --- 2. Download Dataset ---
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("carloss-workspace-ako43").project("rubik-s-cube-sticker-detection-pmtq6")
version = project.version(1)
dataset = version.download("yolov8")

# Required on Windows to prevent multiprocessing errors
if __name__ == "__main__":

    # --- 3. Train ---
    model = YOLO("yolov8n.pt")  # Swap to 'yolov8s.pt' or 'yolov8m.pt' for more accuracy

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,         # Lower to 8 or 4 if you run out of memory
        name="rubiks_cube_detector",
        project="runs",
        device=0,         # Use GPU if available; set to 'cpu' if not
        exist_ok=True,    # Overwrites instead of creating a new numbered folder
    )

    # --- 4. Evaluate ---
    metrics = model.val(split="test")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print("Training complete. Weights saved to runs/detect/rubiks_cube_detector/weights/best.pt")