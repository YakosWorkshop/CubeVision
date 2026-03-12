#!/usr/bin/env bash

set -e

IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"
CONTAINER_NAME="ultralytics_devs"

sudo docker run -it --rm \
	--name "$CONTAINER_NAME" \
	--runtime=nvidia \
	--network=host \
	--ipc=host \
	--privileged \
	--device /dev/video0:/dev/video0 \
	-e DISPLAY="$DISPLAY" \
	-e QT_X11_NO_MITSHM=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v /home/csuser/cs4391_spring26:/ultralytics/cs4391_spring26 \
	-v /tmp/argus_socket:/tmp/argus_socket \
	"$IMAGE"
