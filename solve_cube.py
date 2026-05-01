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

    try:
        solution = kociemba.solve(cube_string)

        print("\nSolution moves:")
        print(solution)

        print("\nStep-by-step:")
        for i, move in enumerate(solution.split(), start=1):
            print(f"{i}. {move}")

    except Exception as exc:
        print("\nSolver failed.")
        print(exc)
        print("This usually means the scanned cube state is not physically valid.")
