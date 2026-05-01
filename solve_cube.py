import argparse
import json

import kociemba

from cube_state import build_kociemba_string, validate_cube_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="cube_faces.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cube_faces = json.load(f)

    valid, message = validate_cube_faces(cube_faces)

    if not valid:
        print("Invalid cube scan:")
        print(message)
        return

    cube_string = build_kociemba_string(cube_faces)

    print("\nKociemba cube string:")
    print(cube_string)
