"""Capture and serialize all six faces of a Rubik's Cube from a webcam."""

import argparse
import json
from pathlib import Path

import cv2 as cv
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from color_utils import classify_color_bgr

FACES = {
    "Up": "white",
    "Right": "orange",
    "Front": "blue",
    "Down": "yellow",
    "Left": "red",
    "Back": "green",
}

FACE_ORDER = ["Up", "Right", "Front", "Down", "Left", "Back"]


def extract_detection(
    result: Results, frame: NDArray
) -> list[dict[str, object]] | None:
    """Convert model predictions into the nine highest-confidence tile detections.

    Bounding boxes are clamped to the frame boundaries, and invalid boxes are
    ignored. For detections whose model class is ``center``, the tile color is
    classified from the corresponding image region. Other detections retain
    their model-provided class label.

    Args:
        result: Prediction result containing bounding boxes, confidences, class
            identifiers, and class names.
        frame: Source BGR image used to classify the center tile's color.

    Returns:
        The nine highest-confidence detections, each containing ``label``,
        ``confidence``, ``cx``, ``cy``, and ``bbox`` fields. Returns ``None``
        when fewer than nine valid detections are available.
    """
    boxes = result.boxes
    detections = []
    height, width = frame.shape[:2]

    for xyxy, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        model_label = result.names[int(class_id)]

        if model_label == "center":
            roi = frame[y1:y2, x1:x2]
            color_label = classify_color_bgr(roi)
        else:
            color_label = model_label

        detections.append(
            {
                "label": color_label,
                "confidence": float(confidence),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "bbox": [x1, y1, x2, y2],
            }
        )

    detections = sorted(detections, key=lambda det: det["confidence"], reverse=True)

    if len(detections) < 9:
        return None

    return detections[:9]


def to_grid(tiles: list[dict]) -> list[list[dict]]:
    """Arrange nine tile detections into a row-major 3x3 grid.

    Tiles are sorted from top to bottom by their center y-coordinate, divided
    into three rows, and then sorted left to right within each row by their
    center x-coordinate. The input list is reordered in place.

    Args:
        tiles: Nine detection dictionaries containing numeric ``cx`` and ``cy``
            coordinates.

    Returns:
        A 3x3 matrix containing the original detection dictionaries.

    Raises:
        ValueError: If ``tiles`` does not contain exactly nine detections.
    """
    if len(tiles) != 9:
        raise ValueError(f"Expected 9 tiles, got {len(tiles)}")

    tiles.sort(key=lambda tile: tile["cy"])

    grid = []

    for i in range(0, 9, 3):
        row = tiles[i : i + 3]
        row.sort(key=lambda tile: tile["cx"])
        grid.append(row)

    return grid


def draw_grid(frame, face_grid, origin=(20, 20), cell_size=100):
    """Draw a labeled representation of a detection grid onto an image.

    Each grid cell is rendered as a white square containing the detection's
    ``label`` value. The supplied frame is modified in place.

    Args:
        frame: Image on which to draw the grid.
        face_grid: Matrix of detection dictionaries with a ``label`` field.
        origin: Pixel coordinates of the grid's top-left corner.
        cell_size: Width and height, in pixels, of each rendered cell.
    """
    x0, y0 = origin

    for row_index, row in enumerate(face_grid):
        for col_index, tile in enumerate(row):
            x1 = x0 + col_index * cell_size
            y1 = y0 + row_index * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                cv.FILLED,
            )
            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                2,
            )
            cv.putText(
                frame,
                str(tile["label"]),
                (x1 + 4, y1 + cell_size // 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                1,
                cv.LINE_AA,
            )


def only_label(face_grid):
    """Extract color labels from a matrix of tile detections.

    Args:
        face_grid: Matrix of detection dictionaries with a ``label`` field.

    Returns:
        A matrix with the same dimensions containing only label values.
    """
    face_labels = []
    for i, row in enumerate(face_grid):
        curr = []
        for j, tile in enumerate(row):
            curr.append(tile["label"])
        face_labels.append(curr)

    return face_labels


def main():
    """Capture and save the six faces of a Rubik's Cube.

    Command-line options select the YOLO model, camera, confidence threshold,
    and output path. Webcam frames are processed until each face in
    ``FACE_ORDER`` has been captured. A face can be saved only when nine tiles
    form a grid and its center label matches the color required by ``FACES``.

    Press Enter to save a valid face, ``R`` to rescan the current face, or ``Q``
    to quit. Captured grids are reduced to color labels and stored under their
    one-letter face keys. The JSON output is written only after all six faces
    have been captured.

    Raises:
        RuntimeError: If the requested camera cannot be opened.
        OSError: If the completed cube data cannot be written to the output
            file.
        TypeError: If the completed cube data cannot be serialized as JSON.
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
    print("Show the faces in this order: White, Orange, Blue, Yellow, Red, Green")
    print("Press ENTER to save the current face.")
    print("Press R to rescan current face.")
    print("Press Q to quit.")

    while current_face_index < len(FACE_ORDER):
        face_name = FACE_ORDER[current_face_index]

        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        annotated = result.plot()

        detections = extract_detection(result, frame)
        is_valid = False

        if detections != None:
            try:
                grid = to_grid(detections)
                draw_grid(annotated, grid)

                center_color = grid[1][1]

                if center_color["label"] != FACES[face_name]:
                    message = (
                        f"{face_name}: current center is '{center_color['label']}', but '{FACES[face_name]}' is required "
                        f"Show a different face."
                    )
                    is_valid = False
                else:
                    message = (
                        f"{face_name}: Ready to capture '{FACES[face_name]}' face "
                        f"({len(cube_faces)}/6 saved). ENTER = save."
                    )
                    is_valid = True
            except ValueError as exc:
                grid = None
                message = f"Grid error: {exc}"
                is_valid = False
        else:
            grid = None
            message = (
                f"{face_name}: Show '{FACES[face_name]}' face ({len(cube_faces)}/6 saved). Need 9 tiles."
            )
            is_valid = False

        cv.putText(
            annotated,
            message,
            (20, annotated.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if is_valid else (0, 0, 255),
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

        if key == ord("\r") and is_valid and grid != None:
            cube_faces[face_name[0]] = only_label(grid)

            print(f"\nSaved {face_name} face with center color '{FACES[face_name]}':")
            for row in cube_faces[face_name[0]]:
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
