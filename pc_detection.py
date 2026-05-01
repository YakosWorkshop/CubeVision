import argparse
from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
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
    Extract detection information from a YOLO detection result object.
    This function processes the detection results from a YOLO model and converts them
    into a standardized dictionary format containing label, confidence score, and
    center coordinates for each detected object.
    
    Args:
        result: A YOLO result object containing boxes with detection information.
                Expected to have attributes: boxes (with xyxy, cls, conf) and names.
                
    Returns:
        list: A list of dictionaries, each containing:
            - label (str): The class name of the detected object
            - confidence (float): The confidence score of the detection (0.0 to 1.0)
            - cx (float): The x-coordinate of the bounding box center
            - cy (float): The y-coordinate of the bounding box center
        Returns an empty list if no boxes are detected or if result.boxes is None.
        
    Raises:
        None: Gracefully handles None boxes by returning an empty list.
    """
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections

    names = result.names
    xyxy_values = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    for xyxy, class_id, confidence in zip(xyxy_values, class_ids, confidences):
        x1, y1, x2, y2 = xyxy
        detections.append(
            {
                "label": names[int(class_id)],
                "confidence": float(confidence),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
    return detections


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
        detections = extract_detections(result)

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
