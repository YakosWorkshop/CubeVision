import argparse
import json
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

from face_grid import build_face_grid
from pc_detection import extract_detections, draw_face_grid


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
