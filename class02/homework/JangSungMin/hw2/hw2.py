#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.runtime import Core
from threading import Lock
from iotdemo import FactoryController

FORCE_STOP = False

    
def thread_cam1(q):
    # TODO: MotionDetector

    # TODO: Load and initialize OpenVINO

    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))
        # TODO: Motion detect

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        # abnormal detect
        
        # TODO: Inference OpenVINO

        # TODO: Calculate ratios

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector

    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        # TODO: Detect motion

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        # TODO: Detect color

        # TODO: Compute ratio
        

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

    # TODO: HW2 Create a Queue
    q = Queue()
    
    t1 = threading.Thread(target=thread_cam1, args=(q, ))
    t2 = threading.Thread(target=thread_cam2, args=(q, ))
    t1.start()
    t2.start()
    
    
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            name, data = q.get()
                
            # TODO: Control actuator, name == 'PUSH'

            
            if name == 'VIDEO:Cam1 live':
                imshow(name,data)
            elif name == 'VIDEO:Cam2 live':
                imshow(name,data)
            elif name == 'DONE':
                FORCE_STOP = True
            

            q.task_done()
            

    cv2.destroyAllWindows()
    t1.join()
    t2.join()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
