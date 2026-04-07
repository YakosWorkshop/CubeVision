import cv2 as cv
import numpy as np

COLOR_ORDER = ["white", "yellow", "red", "orange", "blue", "green"]

def crop_center(roi, frac=0.5):
  h, w, = roi.shape[:2]
  ch, cw = int(h * frac), int(w * frac)
  y1 = max((h - ch) // 2, 0)
  x1 = max((w - cw) // 2, 0)
  return roi[y1:y1+ch, x1:x1+cw]

def mean_hsv(roi)
  center = crop_center(roi, 0.5)
  hsv = cv.cvtColor(center, cv.COLOR_BGR2HSV)
  return hsv.reshape(-1, 3).mean(axis=0)

def classify_color_bgr(roi):
    h, s, v = mean_hsv(roi)

# white: low on saturation but high on value
if s < 40 and v > 120:
  return "white"

# yellow
if 20 <= h <= 40:
  return "yellow"

# orange
if 8 <= h < 20:
  return "orange"

# red (wrapout in HSV)
if h < 8 or h >= 170:
  return "red"

# blue
if 90 <= h <= 130:
  return "blue"

# green
if 45 <= h <= 89:
  return "green"

return "unknown"
