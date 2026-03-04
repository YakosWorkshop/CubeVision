import math

import numpy as np
import cv2 as cv

def order_points_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)
    
    s = pts.sum(axis=1) # x + y
    d = np.diff(pts, axis=1).ravel() # x - y
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    
    return tl, tr, br, bl

def length(p,q):
    return float(np.linalg.norm(p - q))

def angle_deg(p, q, r):
    v1 = p - q
    v2 = r - q
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    cosangle = np.dot(v1, v2) / (n1 * n2)
    cosangle = np.clip(cosangle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cosangle)))

def is_horizontal(p, q, max_slope_deg=10):
    dx = float(q[0] - p[0])
    dy = float(q[1] - p[1])
    
    angle = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9))
    
    return angle <= max_slope_deg

def is_vertical(p, q, max_slope_deg=10):
    dx = float(q[0] - p[0])
    dy = float(q[1] - p[1])
    
    angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9))
    
    return angle <= max_slope_deg

def approx_is_square(approx, side_rel_tol: float = 0.60, angle_tol_deg: int = 20):
    pts = approx.reshape(-1,2).astype(np.float32)
    
    # Shape must have four corners
    if pts.shape[0] != 4:
        return False
    
    # Find the corners
    A, B, C, D = order_points_clockwise(pts)
    
    # Find the length of each side
    AB = length(A, B)
    BC = length(B, C)
    CD = length(C, D)
    DA = length(D, A)
    sides = np.array([AB, BC, CD, DA], dtype=np.float32)
    max_side = sides.max()
    cutoff = max_side * side_rel_tol
    
    # if any side is much smaller than the longest side, return False
    for s in sides:
        if s < cutoff:
            return False
    
    # Corners must be roughly 90 degrees    
    min_angle = 90 - angle_tol_deg
    max_angle = 90 + angle_tol_deg
    
    angle_A = angle_deg(D, A, B)
    angle_B = angle_deg(A, B, C)
    angle_C = angle_deg(B, C, D)
    angle_D = angle_deg(C, D, A)
    angles = [angle_A, angle_B, angle_C, angle_D]
    for a in angles:
        if a < min_angle or a > max_angle:
            return False
    
    # Find the bounding box
    left_boundary = np.min([A[0], B[0], C[0], D[0]])
    right_boundary = np.max([A[0], B[0], C[0], D[0]])
    top_boundary = np.min([A[1], B[1], C[1], D[1]])
    top_left = np.array([left_boundary, top_boundary])
    top_right = np.array([right_boundary, top_boundary])
    
    
    # Is B above A on the y-axis?
    if B[1] < A[1]:
        angle_P = int(angle_deg(A, top_left, B))
        
        if angle_P > angle_tol_deg:
            return False
    else:
        angle_Q = int(angle_deg(A, top_right, B))
        
        if angle_Q > angle_tol_deg:
            return False
        
    return True
    
    
    

def check_quad_constraints(approx, side_rel_tol=0.15, angle_tol_deg=10, hv_tol_deg=10):
    
    pts = approx.reshape(-1, 2).astype(np.float32)
    
    # Shape must have four corners
    if pts.shape[0] != 4:
        return False, {"reason": "not_4_points", "n": pts.shape[0]}

    # Find the each corner
    A, B, C, D = order_points_clockwise(pts)

    # Find the length of each side
    AB = length(A, B)
    BC = length(B, C)
    CD = length(C, D)
    DA = length(D, A)
    sides = np.array([AB, BC, CD, DA], dtype=np.float32)
    mean_side = float(np.mean(sides))
    if mean_side < 1e-6:
        return False, {"reason": "degenerate"}

    # "all four lines roughly same length"
    # Use relative deviation from mean
    side_ok = np.all(np.abs(sides - mean_side) <= side_rel_tol * mean_side)

    # Corner angles (should be ~90)
    angA = angle_deg(D, A, B)
    angB = angle_deg(A, B, C)
    angC = angle_deg(B, C, D)
    angD = angle_deg(C, D, A)
    angles = [angA, angB, angC, angD]
    angle_ok = all(abs(a - 90.0) <= angle_tol_deg for a in angles)

    # Horizontal/Vertical constraints
    # AB and CD horizontal; AD and BC vertical (note: you wrote AC/BC but typical is AD/BC)
    horiz_ok = is_horizontal(A, B, hv_tol_deg) and is_horizontal(C, D, hv_tol_deg)
    vert_ok  = is_vertical(A, D, hv_tol_deg) and is_vertical(B, C, hv_tol_deg)

    ok = bool(side_ok and angle_ok and horiz_ok and vert_ok)

    debug = {
        "A": A.tolist(), "B": B.tolist(), "C": C.tolist(), "D": D.tolist(),
        "sides": {"AB": AB, "BC": BC, "CD": CD, "DA": DA, "mean": mean_side},
        "angles": {"A": angA, "B": angB, "C": angC, "D": angD},
        "checks": {"side_ok": bool(side_ok), "angle_ok": bool(angle_ok),
                   "horiz_ok": bool(horiz_ok), "vert_ok": bool(vert_ok)}
    }
    return ok, debug

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Canny edge detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 20, 40)
    
    # Dilate the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(canny, kernel, iterations=2)
    
    # Find contours
    (contours, hierarchy) = cv.findContours(dilated.copy(), 
                                            cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate contour
        epsilon = 0.03 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        ok = approx_is_square(approx, side_rel_tol=0.20, angle_tol_deg=12)

        if ok:
            cv.polylines(frame, [approx], True, (0,255,0), 2)

        
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
