import time

import cv2
import numpy as np
from labels import labels

import pid
from utils import detect,  draw_detections_on_image, get_center_from_coordinate, draw_predictions_on_image, get_center
import string
import random
from pid import PidController

from kalman_filter import KalmanFilter

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)


def convert(detections):
    dic_detect = {}
    size  = int(detections["num_detections"])
    for ind  in range(size):
        dic_detect[random_str(7)] = {'detection_classes' :detections["detection_classes"][ind],
        "detection_scores":detections["detection_scores"][ind], "detection_boxes":detections["detection_boxes"][ind]}
    return dic_detect


def random_str(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str



class Tracker:

    def __init__(self, window_name, w, h, drone=None, model=None, vision=None):

        self.window_name = window_name
        self.net = model.signatures['serving_default']
        self.frames = vision
        self.h = h
        self.w = w
        self.center_x = int(w/2)
        self.center_y = int(h/2)
        self.center_z = 100
        t = str(time.clock())
        self.drone = drone

    def locate(self, frame):

        (h, w) = frame.shape[:2]

        detections = detect(frame, self.net)
        objects = convert(detections)

        found_persons = []

        # Loop over the detections
        for key, value in objects.items():

            # extract the index of the class label from the 'detections'
            idx = value["detection_classes"]

            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = value["detection_scores"]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if idx == 1 and confidence > 0.5:
              # compute the (x, y)-coordinates of
              # the bounding box for the object
              print(value["detection_boxes"])
              box = value["detection_boxes"] * np.array([w, h, w, h])
              found_persons.append(box)

        return found_persons, detections


    def track(self):

        #cv2.namedWindow(self.window_name,  cv2.WINDOW_FULLSCREEN)

        stop = False

        #Create KalmanFilter object KF
        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        KF = KalmanFilter()
        # set first point, the first location is VERY important

        delta_x = None
        delta_y = None
        delta_z = None



        while not stop:
            # Obtain new frame
            if self.window_name == "webcam":
                t, frame = self.frames.get_webcam()

            else:
                t, frame = self.frames.get_drone()

            if frame is not None:
                (h, w) = frame.shape[:2]
                found_persons, detections = self.locate(frame)
                frame = cv2.circle(frame, (self.center_x, self.center_y), 0, COLOR_GREEN , 20)


                if found_persons:
                    x3, y3, z3 = get_center_from_coordinate(frame, detections=detections)
                    current_measurement = np.array((x3, y3), dtype=np.float32).reshape(2, 1)
                    KF.update(current_measurement)
                    frame = draw_detections_on_image(frame, detections, labels)
                    delta_x = x3 - self.center_x
                    delta_y = y3 - self.center_y
                    delta_z = self.center_z - z3

                    if self.drone.setup == False:
                        self.drone.pid_setup(X0=delta_x, Y0=delta_y, Z0=delta_z)

                else:
                    (x, y) = KF.predict()
                    print("prediction :", x, y)
                    current_prediction = (int(x), int(y))
                    x3, y3 = int(current_prediction[0]), int(current_prediction[1])
                    frame = cv2.circle(frame, (x3, y3), 0, COLOR_RED, 20)


                if delta_z != None:
                        self.drone.autodrive(delta_x=delta_x, delta_y=delta_y, delta_z=delta_z, radius=0)
                cv2.imshow(self.window_name, frame)


                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 :
                    break

        cv2.destroyWindow(self.window_name)
