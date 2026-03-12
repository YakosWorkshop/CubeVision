import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--cam_id", type=int, default=0)
    
    args = parser.parse_args()
    
    model = YOLO(args.model)
    cam = args.cam_id
    
    model.predict(
    source=cam,
    conf=0.5,
    show=True,
    show_labels=True,
    show_conf=True,
    )
    
if __name__ == "__main__":
    main()