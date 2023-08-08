import argparse
import time

import cv2
import numpy as np


import tracker
import keyboard
import pyparrot

from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from pyparrot.Model import Model
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
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from pyparrot.Model import Model
import threading
import cv2
import time


# Initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        # print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "test_image_%06d.png" % self.index
            # uncomment this if you want to write out images every time you get a new one
            #cv2.imwrite(filename, img)
            self.index +=1



# Loads the module from internet, unpacks it and initializes a Tensorflow saved model.
def load_model(model_name):
    model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path('.tmp').absolute()
    )
    model = tf.saved_model.load(model_dir + '/saved_model')

    return model


class Drone(Bebop):
    def __init__(self):
        self.camera = None

    def init_drone(self):
        #check internals parameters
        pass



    def land(self):
        self.safe_land(10)



    def takeoff(self):
        self.safe_takeoff(10)



    def start_camera(self):
        if self.camera == None:
            self.camera = DroneVision(bebop, Model.BEBOP)
        else:
            pass

        success = self.camera.open_video()
            if (success):
                print("camera open")



    def connect(self):
        success = self.connect(5)
        if (success):
            self.network_success = True

    def disconnect(self):
        success = self.disconnect()



    def stop_camera(self):
        self.camera.close_video()



def main():

    # Load our serialized model from disk
    print("[INFO] Loading model...")
    #net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # Initialize the drone and allow the camera sensor to warm up
    drone, status = init_drone()

    frames = frame_grabber(drone)
    time.sleep(2.0)

    drone.takeoff()
    time.sleep(5.0)


    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('videos/tracking' + str(time.clock()) + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 15., (3*h, 3*w))
    track_person = tracker.Tracker("Tracking", drone, net, frames, vmax, w, h, kp, ki, kd, args['confidence'], 80)
    track_person.track(out)
    drone.land()
    out.release()

#main()
model = load_model()
