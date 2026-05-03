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
    """
    Validates that a scanned cube contains all required faces and valid color counts.

    This function performs basic consistency checks before the cube is passed to the
    solver. It verifies that all six faces are present, that each face contains exactly
    9 stickers, that no sticker was classified as "unknown", and that the full cube
    contains exactly six colors with nine occurrences of each color.

    These checks do not guarantee that the cube state is physically solvable, but they
    catch common scanning and classification errors before calling Kociemba's algorithm.

    Args:
        cube_faces (dict): A dictionary mapping cube face names to 3x3 sticker grids.
                           Expected keys are "U", "R", "F", "D", "L", and "B".

    Returns:
        tuple: A pair in the form (is_valid, message), where:
            - is_valid (bool): True if the cube passes the basic validation checks.
            - message (str): "ok" if valid, otherwise an explanation of the failure.
    """
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
    """
    Builds a mapping from detected sticker colors to standard cube face letters.

    Kociemba's solver does not use color names directly. Instead, it expects each
    sticker to be represented by one of the six face letters: U, R, F, D, L, or B.
    Since the center tile of each Rubik's Cube face never moves, the center sticker
    color determines which detected color corresponds to each face letter.

    Args:
        cube_faces (dict): A dictionary mapping face letters to 3x3 sticker grids.
                           Each grid must have a valid center tile at index [1][1].

    Returns:
        dict: A dictionary mapping color labels to face letters.
              Example: {"white": "U", "red": "R", "green": "F", ...}

    Raises:
        ValueError: If two faces have the same center color, which indicates that the
                    same face may have been scanned twice or color classification failed.
    """
    color_to_face = {}

    for face in FACE_ORDER:
        center_color = cube_faces[face][1][1]

        if center_color in color_to_face:
            raise ValueError(f"Duplicate center color detected: {center_color}")

        color_to_face[center_color] = face

    return color_to_face


def build_kociemba_string(cube_faces):
    """
    Converts scanned cube faces into the 54-character string required by Kociemba.

    This function first validates the scanned cube data, then creates a color-to-face
    mapping using the center stickers. It converts each sticker color into its
    corresponding cube face letter and concatenates the result in Kociemba's expected
    face order: U, R, F, D, L, B.

    Args:
        cube_faces (dict): A dictionary containing all six scanned faces as 3x3 grids.
                           Expected format:
                           {
                               "U": [[...], [...], [...]],
                               "R": [[...], [...], [...]],
                               "F": [[...], [...], [...]],
                               "D": [[...], [...], [...]],
                               "L": [[...], [...], [...]],
                               "B": [[...], [...], [...]]
                           }

    Returns:
        str: A 54-character cube string using only the letters U, R, F, D, L, and B.

    Raises:
        ValueError: If cube validation fails.
        ValueError: If duplicate center colors are detected.
        ValueError: If the generated cube string is not exactly 54 characters long.

    Example:
        >>> build_kociemba_string(cube_faces)
        'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    """
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
