#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import sys
import cv2
import numpy as np
# from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector

    # TODO: Load and initialize OpenVINO

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('/home/taejoon/intel-05/class02/smart-factory/resources/conveyor.mp4')
    # while not FORCE_STOP:
    while True:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
       
        # TODO: Motion detect

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        # abnormal detect
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q.put(frame)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

+

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector

    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('/home/taejoon/intel-05/class02/smart-factory/resources/conveyor.mp4')

    # while not FORCE_STOP:
    while True:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q.put(frame)
        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        # TODO: Detect motion

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

        # TODO: Detect color

        # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%")

        # TODO: Enqueue to handle actuator 2

    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    que_1 = Queue()
    que_2 = Queue()
    
    t1 = threading.Thread(target=thread_cam1,args=(que_1,))
    t2 = threading.Thread(target=thread_cam2,args=(que_2,))


    
    t1.start()
    t2.start()
    cam2_motion_detector = MotionDetector()

    cam2_motion_detector.load_preset("/home/taejoon/intel-05/class02/smart-factory/resources/motion.cfg")
    
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    # with FactoryController(args.device) as ctrl:
        # while not FORCE_STOP:
    while True:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            frame_1 = que_1.get()
            frame_2 = que_2.get()
            detect_frame2 = cam2_motion_detector.detect(frame_2)
            
           
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            # numpy_horizontal = np.vstack((frame_1, detect_frame2))
            # cv2.imshow('object_detection', numpy_horizontal)
            # cv2.waitKey(200)
            imshow("frame1", frame_1)
            cv2.waitKey(100)
            imshow("frame_2",detect_frame2)
            # cv2.waitKey()

            # TODO: Control actuator, name == 'PUSH'

            # if name == 'DONE':
            #     FORCE_STOP = True

            # q.task_done()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
