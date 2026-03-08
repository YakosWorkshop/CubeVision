"""
Rubik's Cube Sticker Detection — Laptop Camera Script
Personal use only — for testing inference before deploying to Jetson.
Requires best.pt from runs/detect/rubiks_cube_detector/weights/best.pt
"""

from ultralytics import YOLO

# --- 1. Load Trained Model ---
model = YOLO("runs/detect/runs/detect/rubiks_cube_detector/weights/best.pt")

# --- 2. Run Live Webcam Inference ---
# source=0 is your built-in laptop camera
# Change to source=1 for an external USB camera
model.predict(
    source=0,
    conf=0.5,
    show=True,
    show_labels=True,
    show_conf=True,
)
