import json
from pathlib import Path


def _cluster_sorted_values(values, expected_groups=3):
    """
    Clusters sorted values into groups based on the largest gaps between consecutive values.
    This function takes a list of values and partitions them into a specified number of groups
    by identifying and cutting at the largest gaps. It then calculates the center (mean) of
    each group. Primarily used for organizing detected grid positions into rows/columns.
    Args:
        values (list): A list of numeric values to be clustered.
        expected_groups (int, optional): The expected number of groups to partition the values into.
                                        Defaults to 3. The total number of values must equal
                                        expected_groups squared (e.g., 9 values for 3 groups).
    Returns:
        list: A list of center values (means) for each group, one per group.
    Raises:
        ValueError: If the number of values does not equal expected_groups squared.
        ValueError: If the values cannot be successfully split into the expected number of groups
                   with equal group sizes.
    Example:
        >>> _cluster_sorted_values([1, 2, 3, 10, 11, 12, 20, 21, 22], expected_groups=3)
        [2.0, 11.0, 21.0]
    """
    ordered = sorted(values)
    if len(ordered) != expected_groups ** 2:
        raise ValueError(f"Expected {expected_groups ** 2} values, got {len(ordered)}")

    gaps = []
    for index in range(len(ordered) - 1):
        gap = ordered[index + 1] - ordered[index]
        gaps.append((gap, index))

    cut_indexes = sorted(index for _, index in sorted(gaps, reverse=True)[: expected_groups - 1])

    groups = []
    start = 0
    for cut_index in cut_indexes:
        groups.append(ordered[start:cut_index + 1])
        start = cut_index + 1
    groups.append(ordered[start:])

    if len(groups) != expected_groups or any(len(group) != expected_groups for group in groups):
        raise ValueError("Could not split detections into a 3x3 grid")

    centers = [sum(group) / len(group) for group in groups]
    return centers


def build_face_grid(detections):
    """
    Organizes detected stickers into a 3x3 grid structure representing a single face of a cube.
    This function takes a list of sticker detections and arranges them into a 3x3 grid by:
    1. Clustering detections into 3 rows based on their y-coordinates (cy)
    2. Assigning each detection to its nearest row center
    3. Sorting each row by x-coordinates (cx) from left to right
    4. Extracting the label of each sticker
    Args:
        detections (list): A list of detection dictionaries, each containing:
            - "cy" (float): Center y-coordinate of the sticker
            - "cx" (float): Center x-coordinate of the sticker
            - "label" (str): The label/color identifier of the sticker
            Expected length: exactly 9 detections
    Returns:
        list: A 3x3 nested list where each element is a sticker label:
            [[top_left, top_center, top_right],
             [mid_left, mid_center, mid_right],
             [bot_left, bot_center, bot_right]]
    Raises:
        ValueError: If the number of detections is not exactly 9
        ValueError: If any row does not contain exactly 3 stickers after clustering
    """
    if len(detections) != 9:
        raise ValueError(f"Need exactly 9 detections to build a face, got {len(detections)}")

    row_centers = _cluster_sorted_values([det["cy"] for det in detections])
    for det in detections:
        det["row"] = min(range(3), key=lambda idx: abs(det["cy"] - row_centers[idx]))

    ordered = []
    for row_index in range(3):
        row = [det for det in detections if det["row"] == row_index]
        if len(row) != 3:
            raise ValueError(f"Row {row_index} has {len(row)} stickers instead of 3")

        row.sort(key=lambda det: det["cx"])
        ordered.append([det["label"] for det in row])

    return ordered


def save_face_grid(face_grid, output_path, source_frame=None):
    """
    Save a face grid configuration to a JSON file.
    This function serializes face grid data along with its center point and 
    optional source frame information to a JSON file at the specified output path.
    Parent directories are created automatically if they don't exist.
    Args:
        face_grid: A 2D grid structure representing face data, where face_grid[1][1] 
                   is the center point.
        output_path (str or Path): The file path where the JSON output will be saved.
        source_frame (int, optional): The source frame number associated with this face grid.
                                      If provided, it will be included in the output JSON.
                                      Defaults to None.
    Returns:
        None
    Raises:
        OSError: If parent directories cannot be created or file cannot be written.
        TypeError: If payload cannot be serialized to JSON.
    """
    payload = {
        "face": face_grid,
        "center": face_grid[1][1],
    }
    if source_frame is not None:
        payload["frame"] = int(source_frame)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")