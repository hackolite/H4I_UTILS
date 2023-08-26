import time

import cv2
import numpy as np
from labels import labels

import move_drone
import pid
from utils import detect,  draw_detections_on_image, get_center_from_coordinate, draw_predictions_on_image, get_center
import string
import random


from kalman_filter import KalmanFilter

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)



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



#def get_center(loc):
#  if loc is None:
#    return None
#  return int(loc[0] + (loc[2] - loc[0]) / 2), int(loc[3] + (loc[1] - loc[3]) / 2)


class Tracker:

    def __init__(self, window_name, vmax, w, h, kp, ki, kd, tol, alpha,  drone=None, model=None, vision=None):

        self.window_name = window_name
        self.net = model.signatures['serving_default']
        self.tol = tol
        self.frames = vision
        self.alpha = alpha
        self.h = h
        self.w = w


        t = str(time.clock())

        pidx = pid.PID(vmax, w, h, kp[0], ki[0], kd[0])
        pidy = pid.PID(vmax, w, h, kp[1], ki[1], kd[1])
        pidz = pid.PID(vmax, w, h, kp[2], ki[2], kd[2])
        pidt = pid.PID(vmax, w, h, kp[3], ki[3], kd[3])

        #self.controller = move_drone.MoveDrone(drone, pidx, pidy, pidz, pidt)


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


        #Variable used to control the speed of reading the video
        ControlSpeedVar = 100  #Lowest: 1 - Highest:100
        HiSpeed = 100
        #Create KalmanFilter object KF
        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        debugMode=1
        KF = KalmanFilter()
        # set first point, the first location is VERY important
        KF.update(np.array([[906], [311]], dtype=np.float32))

        while not stop:

            # Obtain new frame
            if self.window_name == "webcam":
                t, frame = self.frames.get_webcam()

            else:
                t, frame = self.frames.get_drone()

            if frame is not None:
                (h, w) = frame.shape[:2]
                found_persons, detections = self.locate(frame)

                if found_persons:
                    x3, y3 = get_center_from_coordinate(frame, detections=detections)
                    current_measurement = np.array((x3, y3), dtype=np.float32).reshape(2, 1)
                    KF.update(current_measurement)
                    frame = draw_detections_on_image(frame, detections, labels)


                else:
                    (x, y) = KF.predict()
                    print("prediction :", x, y)
                    current_prediction = (int(x), int(y))
                    lpx, lpy = int(current_prediction[0]), int(current_prediction[1])
                    frame = cv2.circle(frame, (lpx, lpy), 0, COLOR_RED, 20) # plot prediction dot

                cv2.imshow(self.window_name, frame)


                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 :
                    break

        cv2.destroyWindow(self.window_name)
