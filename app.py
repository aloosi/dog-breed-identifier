import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, redirect
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob

from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/dogdetector")
def dogdetector():
    return render_template('dogdetector.html', image_path="")

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method=="POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)

            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)
            file_extension = f.filename.rsplit('.', 1)[1].lower()
            detected_breeds = []
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                #image = Image.open(io.BytesIO(frame))
                image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                # Perform the detection
                yolo = YOLO('best.pt')
                detections = yolo.predict(image, save=True)
                return display(f.filename)
            
            elif file_extension == 'png':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.png', cv2.UMat(img))[1].tobytes()
                #image = Image.open(io.BytesIO(frame))
                image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                # Perform the detection
                yolo = YOLO('best.pt')
                detections = yolo.predict(image, save=True)
                print(detected_breeds)
                return display(f.filename)
            
            elif file_extension == 'mp4':
                video_path = filepath # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 20.0,(frame_width, frame_height))

                # Initialize the YOLOv8 model
                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # do the YOLOv8 frame detection
                    results = model(frame, save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    # Write the frames to the output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break
                return video_feed()
                
        
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
    return render_template('dogdetector.html', image_path=image_path)

# Display function to show the image or video from the folder_path directory
@app.route('/<path:filename>', methods=["POST"])
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os. path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files [0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()
    environ = request.environ
    if file_extension == 'jpg':
        dest_path = os.path.join('static/images', latest_file)
        shutil.move(filename, dest_path)

        return render_template('dogdetector.html', image_path=dest_path)
        #return send_from_directory(directory, latest_file, environ) #shows the result in seperate tab

    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files) #detected video path

    ok_flag = True
    while ok_flag == True:
        success, image = video.read()
        if cv2.waitKey(0) == 27:
            ok_flag = False
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        frame_data = (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        yield frame_data
        time.sleep(0.1)

# function to display detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(debug=True)