import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data='data.yaml', epochs=100)


# todo: 
# write code to add text files for each image under the labels folder