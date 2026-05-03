import argparse
from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
from color_utils import classify_color_bgr
from face_grid import build_face_grid, save_face_grid


def parse_args():
    """
    Parse and return command-line arguments for the Rubiks cube detection program.
    
    Arguments:
        --model (str): Path to the model file. Required.
        --cam_id (int): Camera ID to use for input. Defaults to 0.
        --conf (float): Confidence threshold for detections. Defaults to 0.5.
        --output (str): Directory path for saving output. Defaults to "saved_faces".
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="saved_faces")
    return parser.parse_args()


def extract_detections(result, frame):
    """
    Extracts and processes YOLO sticker detections from a webcam frame.

    This function converts raw YOLO detection outputs into structured sticker
    detections that can later be organized into a 3x3 Rubik's Cube face grid.
    Each detected sticker bounding box is cropped from the original frame and
    classified using HSV-based color detection.

    The returned detections contain:
        - sticker color label
        - detection confidence
        - sticker center coordinates
        - sticker bounding box coordinates

    Detections are sorted by confidence score and trimmed to the top 9 results,
    since a valid Rubik's Cube face should contain exactly 9 visible stickers.

    Args:
        result: YOLO inference result object containing bounding boxes and
                confidence scores.
        frame: Original webcam frame (numpy array) used for extracting sticker
               regions of interest.

    Returns:
        list: A list of detection dictionaries in the format:
              {
                  "label": str,
                  "confidence": float,
                  "cx": float,
                  "cy": float,
                  "bbox": [x1, y1, x2, y2]
              }

    Example:
        >>> detections = extract_detections(result, frame)
        >>> detections[0]["label"]
        'red'
    """
    detections = []
    boxes = result.boxes

    if boxes is None:
        return detections

    xyxy_values = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    height, width = frame.shape[:2]

    for xyxy, confidence in zip(xyxy_values, confidences):
        x1, y1, x2, y2 = map(int, xyxy)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        color_label = classify_color_bgr(roi)

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

    return detections[:9]


def draw_face_grid(frame, face_grid, origin=(20, 20), cell_size=52):
    """
    Draw a grid of labeled cells on a frame, typically used for face detection visualization.
    
    Args:
        frame: The image frame (numpy array) on which to draw the grid.
        face_grid: A 2D list/array of strings where each element is a label for the corresponding grid cell.
        origin: A tuple (x, y) representing the top-left corner coordinates where the grid starts. Defaults to (20, 20).
        cell_size: The size of each grid cell in pixels (both width and height). Defaults to 52.
    
    Returns:
        None. Modifies the frame in-place by drawing the grid and labels.
    
    Example:
        >>> face_grid = [['A', 'B'], ['C', 'D']]
        >>> draw_face_grid(frame, face_grid, origin=(10, 10), cell_size=50)
    """
    x0, y0 = origin
    for row_index, row in enumerate(face_grid):
        for col_index, label in enumerate(row):
            x1 = x0 + col_index * cell_size
            y1 = y0 + row_index * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv.putText(
                frame,
                label,
                (x1 + 4, y1 + cell_size // 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )


def main():
    args = parse_args()
    model = YOLO(args.model)
    output_dir = Path(args.output)

    cap = cv.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam_id}")

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        detections = extract_detections(result, frame)

        save_message = "Press ENTER to save current face"
        if len(detections) == 9:
            try:
                face_grid = build_face_grid(detections)
                draw_face_grid(annotated, face_grid)
            except ValueError as exc:
                face_grid = None
                save_message = f"Grid error: {exc}"
        else:
            face_grid = None
            save_message = f"Need 9 stickers, found {len(detections)}"

        cv.putText(
            annotated,
            save_message,
            (20, annotated.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if face_grid is not None else (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        cv.imshow("CubeVision", annotated)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord('\r') and face_grid is not None:
            output_path = output_dir / f"face_{frame_index:05d}.json"
            save_face_grid(face_grid, output_path, source_frame=frame_index)
            print(f"Saved face grid to {output_path}")

        frame_index += 1

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
