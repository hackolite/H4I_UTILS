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

import time
from pyparrot.Model import Model
import vlc
import cv2, os


import time

import tracker
from labels import labels



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



class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        pass


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



def detect_objects_on_image(image, model):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)

    # Adding one more dimension since model expect a batch of images.
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)
    print(output_dict)
    num_detections = int(output_dict['num_detections'])
    output_dict = {
        key:value[0, :num_detections].numpy()
        for key,value in output_dict.items()
        if key != 'num_detections'
    }
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def get_coordinate(detections=None, height=None, width=None):
    detection_box = detections['detection_boxes'][0]

    y1 = int(width * detection_box[0])
    x1 = int(height * detection_box[1])
    y2 = int(width * detection_box[2])
    x2 = int(height * detection_box[3])

    return (x1, y1, x2, y2)





def draw_detections_on_image(image, detections, labels, classe=1):
    image_with_detections = image
    width, height, channels = image_with_detections.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    label_padding = 5

    num_detections = detections['num_detections']
    if num_detections > 0:
        for detection_index in range(num_detections):
            if detections['detection_classes'][detection_index]  == 1:
                detection_score = detections['detection_scores'][detection_index]
                detection_box = detections['detection_boxes'][detection_index]
                detection_class = detections['detection_classes'][detection_index]
                detection_label = labels[detection_class]
                detection_label_full = detection_label["name"] + ' ' + str(math.floor(100 * detection_score)) + '%'

                y1 = int(width * detection_box[0])
                x1 = int(height * detection_box[1])
                y2 = int(width * detection_box[2])
                x2 = int(height * detection_box[3])

                # Detection rectangle.
                image_with_detections = cv2.rectangle(
                    image_with_detections,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # Label background.
                label_size = cv2.getTextSize(
                    detection_label_full,
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    2
                )
                image_with_detections = cv2.rectangle(
                    image_with_detections,
                    (x1, y1 - label_size[0][1] - 2 * label_padding),
                    (x1 + label_size[0][0] + 2 * label_padding, y1),
                    color,
                    -1
                )

                # Label text.
                cv2.putText(
                    image_with_detections,
                    detection_label_full,
                    (x1 + label_padding, y1 - label_padding),
                    font,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

    return image_with_detections
