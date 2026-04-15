import argparse
from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
from face_grid import build_face_grid, save_face_grid
from color_utils import COLOR_ORDER

# BGR values for each face color used in the status panel
COLOR_BGR = {
    "white":  (255, 255, 255),
    "yellow": (0,   255, 255),
    "red":    (0,   0,   255),
    "orange": (0,   165, 255),
    "blue":   (255, 0,   0  ),
    "green":  (0,   200, 0  ),
}

TOTAL_FACES = 6


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


def extract_detections(result):
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


def draw_face_status(frame, captured_faces):
    """
    Draw a status panel in the top-right corner showing which of the 6 faces
    have been captured. Each face is shown as a colored dot with the color name.
    Captured faces show a checkmark, missing faces show an X.

    Args:
        frame: The image frame (numpy array) to draw on. Modified in-place.
        captured_faces (dict): Keys are center color strings, values are face grids.

    Returns:
        None.
    """
    panel_x = frame.shape[1] - 160
    panel_y = 20
    line_height = 28

    cv.putText(
        frame,
        f"Faces: {len(captured_faces)}/{TOTAL_FACES}",
        (panel_x, panel_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    for i, color_name in enumerate(COLOR_ORDER):
        y = panel_y + (i + 1) * line_height
        is_captured = color_name in captured_faces
        dot_color = COLOR_BGR.get(color_name, (200, 200, 200))

        # Filled dot for captured, empty circle for missing
        if is_captured:
            cv.circle(frame, (panel_x + 8, y - 5), 7, dot_color, -1)
            symbol = "OK"
            text_color = (0, 255, 0)
        else:
            cv.circle(frame, (panel_x + 8, y - 5), 7, dot_color, 2)
            symbol = "--"
            text_color = (100, 100, 100)

        cv.putText(
            frame,
            f"{symbol} {color_name}",
            (panel_x + 20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv.LINE_AA,
        )


def draw_flash_message(frame, message, color):
    """
    Draw a centered flash message near the top of the frame.
    Used for brief save confirmations and duplicate warnings.

    Args:
        frame: The image frame (numpy array) to draw on. Modified in-place.
        message (str): The message text to display.
        color (tuple): BGR color tuple for the message text.

    Returns:
        None.
    """
    text_size = cv.getTextSize(message, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 60

    # Dark background rectangle for readability
    cv.rectangle(
        frame,
        (text_x - 10, text_y - 25),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )
    cv.putText(
        frame,
        message,
        (text_x, text_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv.LINE_AA,
    )


def main():
    args = parse_args()
    model = YOLO(args.model)
    output_dir = Path(args.output)

    cap = cv.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam_id}")

    # Tracks saved faces by center color e.g. {"white": [[...],[...],[...]], ...}
    captured_faces = {}

    # Flash message state: (message_string, color_bgr, frames_remaining)
    flash = None
    FLASH_DURATION = 60  # frames (~2 seconds at 30fps)

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

        # --- Build face grid if exactly 9 stickers detected ---
        save_message = "Press ENTER to save current face"
        face_grid = None

        if len(captured_faces) == TOTAL_FACES:
            save_message = "All 6 faces captured! Ready to solve."
        elif len(detections) == 9:
            try:
                face_grid = build_face_grid(detections)
                center_color = face_grid[1][1]

                if center_color == "center":
                    save_message = "Center color unclear — adjust angle or lighting"
                    face_grid = None  # Block saving
                elif center_color in captured_faces:
                    save_message = f"'{center_color}' face already saved — show a new face"
                    face_grid = None  # Block saving a duplicate
                else:
                    draw_face_grid(annotated, face_grid)
                    save_message = f"Press ENTER to save '{center_color}' face"
                    
            except ValueError as exc:
                face_grid = None
                save_message = f"Grid error: {exc}"
        else:
            save_message = f"Need 9 stickers, found {len(detections)}"

        # --- Draw overlays ---
        draw_face_status(annotated, captured_faces)

        # Bottom status bar
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

        # Flash message overlay
        if flash is not None:
            msg, color, remaining = flash
            draw_flash_message(annotated, msg, color)
            flash = (msg, color, remaining - 1) if remaining > 1 else None

        cv.imshow("CubeVision", annotated)

        # --- Key handling ---
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 13 and face_grid is not None:  # Enter key
            center_color = face_grid[1][1]

            if center_color in captured_faces:
                # Should not reach here due to face_grid=None above, but safety check
                flash = (f"'{center_color}' already captured!", (0, 0, 255), FLASH_DURATION)
            else:
                output_path = output_dir / f"face_{center_color}.json"
                save_face_grid(face_grid, output_path, source_frame=frame_index)
                captured_faces[center_color] = face_grid
                print(f"Saved '{center_color}' face to {output_path}")

                if len(captured_faces) == TOTAL_FACES:
                    flash = ("All 6 faces captured!", (0, 255, 0), FLASH_DURATION)
                    print("All 6 faces captured. Ready to solve.")
                else:
                    remaining = TOTAL_FACES - len(captured_faces)
                    flash = (f"Saved '{center_color}'! {remaining} face(s) left", (0, 255, 0), FLASH_DURATION)
        frame_index += 1

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()