import argparse
import json
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO


from face_grid import build_face_grid
from pc_detection import extract_detections, draw_face_grid


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def main():
    """
    Captures all six faces of a Rubik's Cube using a webcam and YOLO sticker detection.

    This function guides user through scanning cube faces in a fixed order:
    U, R, F, D, L, B. For each face, the program waits until exactly 9 stickers are
    detected, organizes those detections into a 3x3 grid, and allows the user to save
    the face. The center sticker color is used to prevent the same face from being
    captured more than once, since the center tile of a Rubik's Cube face never changes.

    The final output is a JSON file containing all six captured face grids. This file
    can then be passed into the cube-state conversion and Kociemba solving pipeline.

    Args:
        None directly. Command-line arguments are parsed inside the function:
            --model (str): Path to the YOLO model weights. Defaults to "best.pt".
            --cam_id (int): Camera device index. Defaults to 0.
            --conf (float): YOLO detection confidence threshold. Defaults to 0.5.
            --output (str): Output JSON file for the captured cube faces.
                            Defaults to "cube_faces.json".

    Returns:
        None

    Raises:
        RuntimeError: If the specified camera cannot be opened.
        OSError: If the output JSON file cannot be written.
        TypeError: If the cube face data cannot be serialized to JSON.
    """
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
    
    while current_face_index < len(FACE_ORDER):
        face_name = FACE_ORDER[current_face_index]

        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=args.conf, verbose=False)
        result = results[0]
        annotated = result.plot()

        detections = extract_detections(result, frame)

        if len(detections) == 9:
            try:
                face_grid = build_face_grid(detections)
                draw_face_grid(annotated, face_grid)

                center_color = face_grid[1][1]

                if center_color in captured_center_colors:
                    message = (
                        f"{face_name}: center '{center_color}' already captured. "
                        f"Show a different face."
                    )
                    valid_face = False
                else:
                    message = (
                        f"Show {face_name} face "
                        f"({len(cube_faces)}/6 saved). ENTER = save."
                    )
                    valid_face = True

            except ValueError as exc:
                face_grid = None
                message = f"Grid error: {exc}"
                valid_face = False
        else:
            face_grid = None
            message = (
                f"Show {face_name} face ({len(cube_faces)}/6 saved). "
                f"Need 9 stickers, found {len(detections)}."
            )
            valid_face = False

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
        
        cv.imshow("Cube Capture", annotated)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            print(f"Rescanning {face_name} face.")
            continue

        if key == ord("\r") and valid_face:
            center_color = face_grid[1][1]

            cube_faces[face_name] = face_grid
            captured_center_colors.add(center_color)

            print(f"\nSaved {face_name} face with center color '{center_color}':")
            for row in face_grid:
                print(row)

            print(f"Progress: {len(cube_faces)}/6 faces captured\n")

            current_face_index += 1

    cap.release()
    cv.destroyAllWindows()

    if len(cube_faces) == 6:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(cube_faces, indent=2), encoding="utf-8")
        print(f"Saved full cube to {output_path}")
    else:
        print("Cube capture incomplete. No full cube file saved.")


if __name__ == "__main__":
    main()
