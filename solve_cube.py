import argparse

import kociemba

from common import load_json, CENTER_COLOR_TO_FACE


def cube_json_to_kociemba_string(cube):
    # Input cube format:
    # {
    #   "U": ["white", ... 9 entries],
    #   ...
    # }
    #
    # Kociemba expects a 54-character string in URFDLB face order,
    # where each sticker is mapped to the face letter of its center color.

    face_order = ["U", "R", "F", "D", "L", "B"]

    center_to_face = {}
    for face in face_order:
        center_color = cube[face][4]
        center_to_face[center_color] = face

    chars = []
    for face in face_order:
        for color in cube[face]:
            if color not in center_to_face:
                raise ValueError(f"Color '{color}' not found among center stickers")
            chars.append(center_to_face[color])

    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube_state", type=str, required=True)
    args = parser.parse_args()

    cube = load_json(args.cube_state)
    cube_str = cube_json_to_kociemba_string(cube)
    print("Kociemba string:", cube_str)

    try:
        solution = kociemba.solve(cube_str)
        print("Solution:", solution)
    except Exception as e:
        print("Solver error:", e)
        print("Likely causes:")
        print("- one or more stickers misclassified")
        print("- face order mismatch during capture")
        print("- cube centers do not match the assumed orientation")


if __name__ == "__main__":
    main()
