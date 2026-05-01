import argparse
import json
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

from face_grid import build_face_grid
from pc_detection import extract_detections, draw_face_grid


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="cube_faces.json")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv.VideoCapture(args.cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam_id}")

    cube_faces = {}
    captured_center_colors = set()
    current_face_index = 0

    print("Cube capture started.")
    print("Show the faces in this order: U, R, F, D, L, B")
    print("Press ENTER to save the current face.")
    print("Press R to rescan current face.")
    print("Press Q to quit.")
