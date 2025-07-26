#!/bin/bash          
PROJECT_DIR="/your/path/to/VGGT-Long"
DATASET_DIR="/mnt/raid/dataset"

sudo docker remove -f vggt-long

echo "mount projects: $PROJECT_DIR --> vggt-long:/home"
echo "mount datasets: $DATASET_DIR --> vggt-long:/media"

sudo docker run --name vggt-long -t -d --gpus 'all,"capabilities=compute,utility,graphics"' \
    --volume="$DATASET_DIR:/media:rw" \
    --volume="$PROJECT_DIR:/home:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MINTSHM=1 \
    vggt-long:latest

xhost -si:localuser:root
xhost si:localuser:root

sudo docker exec -it vggt-long /bin/bash -c "rm -rf /home/VGGT-Long/weights && ln -s /workspace/VGGT-Long/weights /home/VGGT-Long/weights"
sudo docker exec -it -w /home/VGGT-Long vggt-long /bin/bash --login