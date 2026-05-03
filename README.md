# Guide to setup CubeVision on Jetson Nano

## 1. clone repo
```bash 
git clone https://github.com/YakosWorkshop/CubeVision.git

cd CubeVision
```

## 2. execute run_ultralytics_jetson.sh

### IMPORTANT: -v /home/csuser/cs4391_spring26:/ultralytics/cs4391_spring26 will have 
### to be edited to the directory of CubeVision in inside run_ultralytics_jetson.sh
### Example: -v /home/username/path/CubeVision:/ultralytics/CubeVision
 	
### it will request password in order to execute
```bash
sh ./run_ultralytics_jetson.sh
```
 
## 3. update/install dependencies

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python==4.11.0.86
pip install kociemba
apt-get update
apt-get install -y libgtk2.0-dev pkg-config libgl1-mesa-glx libglib2.0-0
```

# Running the Full Cube Solver Pipeline

## 4. capture all six cube faces
```bash 
python capture_cube.py --model best.pt
```

### Show the cube faces in this order:
U -> R -> F -> D -> L -> B

### Controls:
ENTER -> save current face

R -> rescan current face

Q -> quit application

### If the camera does not open:
```bash 
python capture_cube.py --model best.pt --cam_id 1
```

### Notes:
- the program waits until exactly 9 stickers are detected before allowing a face to be saved
- the program tracks how many faces have already been captured
- duplicate faces are automatically rejected using the center tile color
- all six captured faces are saved into cube_faces.json

## 5. compute the Rubik's Cube solution
```bash 
python solve_cube.py --input cube_faces.json
```

### The program will:
- validate the cube state
- convert sticker colors into Kociemba notation
- compute a valid Rubik's Cube solution sequence

### Example output:
R U R' U' F2 L D2

## 6. recommended full run order

```bash
python jetson_detection.py --model best.pt
python capture_cube.py --model best.pt
python solve_cube.py --input cube_faces.json 
```
