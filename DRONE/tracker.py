import time

import cv2
import numpy as np
from labels import labels
import kalman_filter
import move_drone
import pid
from utils import detect_objects_on_image,  draw_detections_on_image, get_coordinate
import string
import random



def convert(detections):
    dic_detect = {}
    size  = int(detections["num_detections"])
    for ind  in range(size):
        dic_detect[random_str(7)] = {'detection_classes' :detections["detection_classes"][ind], "detection_scores":detections["detection_scores"][ind], "detection_boxes":detections["detection_boxes"][ind]}
    #print(dic_detect)
    return dic_detect


def random_str(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str




class Tracker:

    def __init__(self, window_name, vmax, w, h, kp, ki, kd, tol, alpha,  drone=None, model=None, vision=None):

        self.window_name = window_name
        self.net = model.signatures['serving_default']
        self.tol = tol
        self.frames = vision
        self.alpha = alpha
        self.kalman = self.init_kalman_filter(4, 4)
        self.h = h

        t = str(time.clock())

        # Files used to create the Kalman filter graphics
        self.file_tracker = open("textfiles/file_tracker" + t + ".txt", "w")
        self.file_time = open("textfiles/file_time" + t + ".txt", "w")
        self.file_kalman = open("textfiles/file_kalman" + t + ".txt", "w")

        pidx = pid.PID(vmax, w, h, kp[0], ki[0], kd[0])
        pidy = pid.PID(vmax, w, h, kp[1], ki[1], kd[1])
        pidz = pid.PID(vmax, w, h, kp[2], ki[2], kd[2])
        pidt = pid.PID(vmax, w, h, kp[3], ki[3], kd[3])

        #self.controller = move_drone.MoveDrone(drone, pidx, pidy, pidz, pidt)

    def init_kalman_filter(self, ax = 5, ay = 5, rx = 10, ry = 10, px0 = 20, py0 = 20, pu0 = 20, pv0 = 20):

        return kalman_filter.KalmanFilterPixel(ax, ay, rx, ry, px0, py0, pu0, pv0)



    def locate(self, frame):

        (h, w) = frame.shape[:2]

        detections = detect_objects_on_image(frame, self.net)
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

    def update_frame(self, frame, tl, br):

        croppedX = max(tl[0] - self.alpha, 0)
        croppedY = max(tl[1] - self.alpha, 0)

        cropped_image = frame[croppedY: br[1] + self.alpha, croppedX:br[0] + self.alpha]

        found_persons, detections = self.locate(cropped_image)

        if found_persons:
            (h, w) = cropped_image.shape[:2]
            found_persons, detections = self.locate(cropped_image)
            #frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
            (x1, y1, x2, y2) = get_coordinate(detections=detections, height=h, width=w)
            if found_persons:
                rect = np.array([x1 + croppedX, y1 + croppedY, x2 - x1, y2 - y1])


            return rect

        return None

    def track_object(self, frame, rect):

        # Create tracker
        tracker = cv2.TrackerKCF_create()

        tracking_height = min(rect[3], self.h*0.8)
        # Initialize tracker with frame and bounding box
        ok = tracker.init(frame, tuple(rect))

        failure = 0

        t0 = time.time()
        t1 = t0

        print("Tracking height: ", tracking_height)
        height = tracking_height

        prev_ex, prev_ey, prev_ez = 0, 0, 0

        while True:

            # Read a new frame
            t, frame = self.frames.get_latest()
            if frame is None:
                t, frame = self.frames.get_webcam()

            if frame is not None:
                # Start timer
                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(frame)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

                elapsed_bbox = time.time() - t0

                elapsed_frame = time.time() - t1

                if ok:
                    # Tracking success
                    tl = (int(bbox[0]), int(bbox[1]))
                    br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                    # Every half second recalculate the person's bounding box
                    # using the person's computed bounding box enlarged a certain alpha
                    # to obtain a bigger bounding box if the person is moving forwards
                    # or a smaller one if the person is moving backwards

                    if elapsed_bbox > 0.5:

                        rect = self.update_frame(frame, tl, br)

                        if rect is not None:

                            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2] + rect[0], rect[3] + rect[1]), (0,255,0), 2, 1)

                            # Recreate tracker
                            tracker = cv2.TrackerKCF_create()

                            # Initialize tracker with frame and bounding box
                            ok = tracker.init(frame, tuple(rect))

                            height = rect[3]

                            print("HEIGHT: ", height)

                        t0 = time.time()

                    #if elapsed_frame > 0.05:

                    px = tl[0] + int(br[0] - tl[0])/2
                    py = tl[1] + int(br[1] - tl[1])/2

                    self.file_tracker.write(str((px, py)) + '\n')

                    self.file_time.write(str(t) + '\n')
                    px_filtered, py_filtered = np.round(self.kalman.filter_pixel((px, py), t)[:2]).astype(np.int)
                    self.file_kalman.write(str((px_filtered, py_filtered)) + '\n')

                    tl_filtered = (tl[0] + px_filtered - px, tl[1] + py_filtered - py)
                    br_filtered = (br[0] + px_filtered - px, br[1] + py_filtered - py)

                    # If the frame vertical center is smaller than the x coordinate of the
                    # observation, the error has to be positive for the drone to go right.
                    ex = px_filtered - frame.shape[1]/2
                    # If the frame horizontal center is bigger than the y coordinate of the
                    # observation, the error has to be positive for the drone to go up.
                    ey = frame.shape[0]/1.8 - py_filtered

                    #self.controller.set_velocities(ex, prev_ex, ey, prev_ey, tracking_height - height, prev_ez, 0, 0)
                    #self.controller.move_drone()

                    prev_ez = tracking_height - height
                    prev_ex = ex
                    prev_ey = ey
                    t1 = time.time()
                    print("FRAME_0 :", tl, br)
                    # Tracking bounding box
                    cv2.rectangle(frame, tl, br, (255,0,0), 2, 1)
                    # Tracking filtered bounding box
                    try:
                        print("FRAME_1 :", tl_filtered, br_filtered)
                        cv2.rectangle(frame, float(tl_filtered), float(br_filtered), (0,0,255), 2, 1)
                    except Exception as e:
                        print("FRAME_2 :", tl_filtered, br_filtered)
                        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                else :
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2)

                    ##self.controller.stop_drone()
                    failure += 1
                    print(failure)
                    #if failure > 50:
                    #    return False

                # Display tracker type on frame
                cv2.putText(frame, "KCF Tracker", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50),2);

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2);

                # Display result
                frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, frame)
                #if out is not None:
                #    out.write(frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

        self.file_tracker.close()
        self.file_kalman.close()
        self.file_time.close()
        return True

    def track(self):

        #cv2.namedWindow(self.window_name,  cv2.WINDOW_FULLSCREEN)

        stop = False

        while not stop:

            # Obtain new frame
            if self.window_name == "Webcam":

                    t, frame = self.frames.get_webcam()

            else:
                t, frame = self.frames.get_latest()



            if frame is not None:
                (h, w) = frame.shape[:2]
                found_persons, detections = self.locate(frame)
                #frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                (x1, y1, x2, y2) = get_coordinate(detections=detections, height=h, width=w)
                if found_persons:
                    # Detection bounding box
                    frame = draw_detections_on_image(frame, detections, labels)

                rect = np.array([x1, y1, x2 - x1, y2 - y1])
                stop = self.track_object(frame, rect)
                cv2.imshow(self.window_name, frame)

                #######stop = self.track_object(frame,  rect)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

        cv2.destroyWindow(self.window_name)
