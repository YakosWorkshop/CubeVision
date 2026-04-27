# RUNNING INSTRUCTIONS
# python capture_cube.py --model best.pt
# run with best.pt in directory or full path instead

import argparse
import json
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

# build_face_grid: organizes 9 detected stickers into a 3x3 grid
# extract_detections: converts YOLO outputs into structured sticker detections
# draw_face_grid: overlays the 3x3 grid + labels on the frame for visualization
from face_grid import build_face_grid
from pc_detection import extract_detections, draw_face_grid


# Standard cube face order used throughout the pipeline.
# This order MUST remain consistent for correct cube reconstruction later.
FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def main():
    """
    Main entry point for capturing all 6 faces of a Rubik's Cube using a webcam.

    Pipeline:
    Camera Frame → YOLO Detection → Sticker Extraction → 3x3 Grid Construction
    → Save each face → Output full cube state (JSON)
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model weights")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera device index")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="cube_faces.json", help="Output file for cube state")
    args = parser.parse_args()

    # Initialize YOLO model and camera
    model = YOLO(args.model)
    cap = cv.VideoCapture(args.cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam_id}")

    # Dictionary to store all captured faces
    cube_faces = {}

    # Index to track which face we are currently capturing
    current_face_index = 0

    # User instructions
    print("Cube capture started.")
    print("Press ENTER to save the current face.")
    print("Press R to rescan current face.")
    print("Press Q to quit.")

    # Main capture loop
    while current_face_index < len(FACE_ORDER):
        face_name = FACE_ORDER[current_face_index]

        # Capture a frame from the camera
        ok, frame = cap.read()
        if not ok:
            break

        # Run YOLO inference
        results = model.predict(source=frame, conf=args.conf, verbose=False)
        result = results[0]

        # Draw YOLO bounding boxes for visualization
        annotated = result.plot()

        # Extract structured detections:
        # Each detection contains bbox, center, confidence, and color label
        detections = extract_detections(result, frame)

        # Attempt to form a valid 3x3 face
        if len(detections) == 9:
            try:
                # Convert detections into ordered 3x3 grid
                face_grid = build_face_grid(detections)

                # Draw grid and labels on frame
                draw_face_grid(annotated, face_grid)

                message = f"Show {face_name} face. ENTER = save."
                valid_face = True

            except ValueError as exc:
                # Grid construction failed (bad ordering / geometry)
                face_grid = None
                message = f"Grid error: {exc}"
                valid_face = False
        else:
            # Not enough or too many detections for a valid face
            face_grid = None
            message = f"Show {face_name} face. Need 9 stickers, found {len(detections)}."
            valid_face = False

        # Display status message
        cv.putText(
            annotated,
            message,
            (20, annotated.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if valid_face else (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        # Show annotated frame
        cv.imshow("Cube Capture", annotated)

        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            # Quit early
            break

        if key == ord("r"):
            # Rescan current face without advancing
            print(f"Rescanning {face_name} face.")
            continue

        if key == ord("\r") and valid_face:
            # Save current face and move to next
            cube_faces[face_name] = face_grid
            print(f"Saved {face_name}: {face_grid}")
            current_face_index += 1

    # Cleanup resources
    cap.release()
    cv.destroyAllWindows()

    # Save final cube state
    if len(cube_faces) == 6:
        output_path = Path(args.output)

        # Save cube state as JSON:
        # { "U": [...], "R": [...], ..., "B": [...] }
        output_path.write_text(json.dumps(cube_faces, indent=2), encoding="utf-8")

        print(f"Saved full cube to {output_path}")
    else:
        print("Cube capture incomplete. No full cube file saved.")


if __name__ == "__main__":
    main()
