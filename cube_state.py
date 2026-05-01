from collections import Counter

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def flatten_grid(grid):
    """
    Converts a 3x3 face grid into a single flat list of sticker labels.

    This function takes a nested list representing one face of the Rubik's Cube
    and flattens it in row-major order. The resulting list is easier to validate,
    count, and convert into the 54-character cube string required by Kociemba's
    solving algorithm.

    Args:
        grid (list): A 3x3 nested list representing one cube face:
            [[top_left, top_center, top_right],
             [mid_left, mid_center, mid_right],
             [bot_left, bot_center, bot_right]]

    Returns:
        list: A flat list containing the 9 sticker labels in row-major order.

    Example:
        >>> flatten_grid([["white", "white", "red"],
        ...               ["blue", "white", "green"],
        ...               ["orange", "yellow", "red"]])
        ["white", "white", "red", "blue", "white", "green", "orange", "yellow", "red"]
    """
    return [cell for row in grid for cell in row]


def validate_cube_faces(cube_faces):
    # Make sure all six faces exist
    for face in FACE_ORDER:
        if face not in cube_faces:
            return False, f"Missing face {face}"

        stickers = flatten_grid(cube_faces[face])

        if len(stickers) != 9:
            return False, f"Face {face} does not have 9 stickers"

        if "unknown" in stickers:
            return False, f"Face {face} contains unknown color"

    # Make sure the full cube has 54 stickers total
    all_colors = []
    for face in FACE_ORDER:
        all_colors.extend(flatten_grid(cube_faces[face]))

    counts = Counter(all_colors)

    # A valid cube should have exactly 6 colors
    if len(counts) != 6:
        return False, f"Expected 6 colors, got {len(counts)}: {counts}"

    # Each color should appear exactly 9 times
    for color, count in counts.items():
        if count != 9:
            return False, f"Color {color} appears {count} times, expected 9"

    return True, "ok"


def build_color_to_face_map(cube_faces):
    color_to_face = {}

    # The center sticker of each face determines that face's color identity
    for face in FACE_ORDER:
        center_color = cube_faces[face][1][1]

        if center_color in color_to_face:
            raise ValueError(f"Duplicate center color detected: {center_color}")

        color_to_face[center_color] = face

    return color_to_face


def build_kociemba_string(cube_faces):
    valid, message = validate_cube_faces(cube_faces)

    if not valid:
        raise ValueError(message)

    color_to_face = build_color_to_face_map(cube_faces)

    cube_string = ""

    # Kociemba expects the order U, R, F, D, L, B
    for face in FACE_ORDER:
        stickers = flatten_grid(cube_faces[face])

        for color in stickers:
            cube_string += color_to_face[color]

    if len(cube_string) != 54:
        raise ValueError(f"Cube string length is {len(cube_string)}, expected 54")

    return cube_string
