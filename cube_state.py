from collections import Counter

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def flatten_grid(grid):
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

