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


import tracker
from labels import labels


def get_center(bbox=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    x3 = ((x2-x1)/2+x1)
    y3 = ((y2-y1)/2+y1)
    return x3,y3

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



def detect(image=None, model=None, detection_type="tensorflow2"):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)

    # Adding one more dimension since model expect a batch of images.
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
    output_dict['detection_boxes'] = output_dict['detection_boxes'].tolist()
    return output_dict



def get_center_from_coordinate(image, detections, classe=1):
        width, height, channels = image.shape
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
                    x3, y3 = get_center([x1,y1,x2,y2])
                    return x3, y3, y2-y1

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


                x3, y3 = get_center([x1,y1,x2,y2])
                image_with_detections = cv2.circle(image_with_detections,(int(x3), int(y3)), 10, (0, 191, 255), 2)

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



def detect_face(face):
    face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
    face_rects = face_cascade.detectMultiScale(face, 1.3, 5)
    return face_rects

def draw_predictions_on_image(image=None, center=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    label_padding = 5
    frame = cv2.circle(image, (int(center[0]), int(center[1])), 10, (0, 0, 255), 2)
    return frame
