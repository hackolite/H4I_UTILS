import argparse
import time

import cv2
import numpy as np
from PIL import Image

import tracker
import pyparrot

from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pathlib
import cv2
import math

from PIL import Image
from google.protobuf import text_format
import platform
import os

from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time
from pyparrot.Model import Model
import vlc
import cv2, os
import cv2,os

import time

import tracker
from utils import detect
#from protos import string_int_label_map_pb2
import string
import random


vmax = 0.4
w, h = 180, 320
#kpx, kdx = 0.0025, 0.05

kpx, kdx, kix = 0., 0., 0.
kpx, kdx, kix = 0.002, 0.04, 0.
kpy, kdy, kiy = 0.004, 0., 0.
kpz, kdz, kiz = 0., 0., 0.
kpz, kdz, kiz = 0.005, 0.05, 0.00001
kpt, kdt, kit = 0., 0., 0.

kp = [kpx, kpy, kpz, kpt]
kd = [kdx, kdy, kdz, kdt]
ki = [kix, kiy, kiz, kit]



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='file;rtp;udp'

# Initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def demo_user_code_after_vision_opened(bebopVision, args):
    pass

def random_str(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str


# Loads the module from internet, unpacks it and initializes a Tensorflow saved model.
def load_model(model_name):
    model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path('.tmp').absolute()
    )
    model = tf.saved_model.load(export_dir=model_dir + '/saved_model', tags=None)

    return model

def load_labels(labels_name):
    labels_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/' + labels_name

    labels_path = tf.keras.utils.get_file(
        fname=labels_name,
        origin=labels_url,
        cache_dir=pathlib.Path('.tmp').absolute()
    )

    labels_file = open(labels_path, 'r')
    labels_string = labels_file.read()

    labels_map = string_int_label_map_pb2.StringIntLabelMap()

    try:
        text_format.Merge(labels_string, labels_map)
    except text_format.ParseError:
        labels_map.ParseFromString(labels_string)

    labels_dict = {}
    for item in labels_map.item:
        labels_dict[item.id] = item.display_name

    return labels_dict


def detect_objects_on_image(image, model):
    input_tensor = tf.convert_to_tensor(image)
    # Adding one more dimension since model expect a batch of images.
    input_tensor = data.astype(np.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict['num_detections'])
    output_dict = {
        key:value[0, :num_detections].numpy()
        for key,value in output_dict.items()
        if key != 'num_detections'
    }
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict



class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
            pass


    def print_toto(self, args):
                print("toto")



class Drone(Bebop):
    def __init__(self):
        Bebop.__init__(self, drone_type="Bebop2")
        self.camera = None

    def init_drone(self):
        #check internals parameters
        pass


    def land(self):
        self.safe_land(10)



    def takeoff(self):
        self.safe_takeoff(10)



    def disconnect(self):
        success = self.disconnect()



    def stop_camera(self):
        self.camera.close_video()



    def open_video(self):
        pass


class Vision(DroneVision):
    def __init__(self, drone, vision):
        DroneVision.__init__(self, drone, vision)
        self.start_time = time.time()
        self.vidcap = cv2.VideoCapture(0)




    def get_drone(self):
        #check internals parameters
        current = time.time()
        start = self.start_time
        return current-start , self.get_latest_valid_picture()



    def get_webcam(self):
        #check internals parameters
        if self.vidcap.isOpened():
            ret, frame = self.vidcap.read()
        else:
            print("cannot open camera")
        current = time.time()
        start = self.start_time
        return current-start, frame


def convert(detections):
    dic_detect = {}
    size  = int(detections["num_detections"])
    for ind  in range(size):
        dic_detect[random_str(7)] = {'detection_classes' :detections["detection_classes"][ind],
        "detection_scores":detections["detection_scores"][ind], "detection_boxes":detections["detection_boxes"][ind]}
    return dic_detect



def is_target_detected(detections=None, classe=None):
    dictionnary = convert(detections)
    for key, value in dictionnary.items():
        if value["detection_classes"] == classe:
            return  {key:value}
    return False


def main(vision="webcam"):
    # Load our serialized model from disk
    print("[INFO] Loading model...")
    model = load_model("ssdlite_mobilenet_v2_coco_2018_05_09")
    print("[INFO] Loading successfull...")


    if vision == "drone":
        # start up the video
        drone = Drone()
        success = drone.connect(5)
        drone.set_video_stream_mode(mode = 'low_latency')

        bebopVision = Vision(drone, Model.BEBOP)
        userVision = UserVision(bebopVision)
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        success = bebopVision.open_video()
        track_person = tracker.Tracker("Tracking", vmax, w, h, kp, ki, kd, 0.8, 80, drone=drone, model=model, vision=bebopVision )
        track_person.track()

    elif vision == "webcam":
        print("... USE WEBCAM ...")
        drone = Drone()
        webcamVision = Vision(drone, Model.BEBOP)
        userVision = UserVision(webcamVision)
        webcamVision.set_user_callback_function(userVision.print_toto, user_callback_args=None)

        while True :
            elapsed_time, frame = webcamVision.get_webcam()
            detections = detect(image=frame, model=model.signatures['serving_default'])
            print(elapsed_time, detections)
            resp = is_target_detected(detections=detections, classe=1)
            if resp:
                track_person = tracker.Tracker("webcam", vmax, w, h, kp, ki, kd, 0.8, 80, drone=None, model=model, vision=webcamVision)
                track_person.track()

            if elapsed_time > 20:
                print("OBJECT NOT DETECTED")
                exit(0)

        else:
            pass


main(vision="webcam")
