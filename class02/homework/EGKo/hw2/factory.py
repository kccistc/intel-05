#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.runtime import Core

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False

def thread_cam1(q : Queue):
    cam1_motion = MotionDetector()
    cam1_motion.load_preset("./resources/motion.cfg")
    # TODO: MotionDetector
    
    # TODO: Load and initialize OpenVINO

    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        q.put(("VIDEO:Cam1 live", frame))
        
        # TODO: Motion detect

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q : Queue):
    # TODO: MotionDetector

    # TODO: ColorDetector

    cap = cv2.VideoCapture("./resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        q.put(("VIDEO:Cam2 live", frame))
        
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

    q = Queue()
    
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()
    
    try:
        with FactoryController(args.device) as ctrl:
            while not FORCE_STOP:
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
                
                name, frame = q.get()
                
                if name == "VIDEO:Cam1 live":
                    imshow(name, frame)
                elif name == "VIDEO:Cam2 live":
                    imshow(name, frame)
                
                # TODO: Control actuator, name == 'PUSH'

                # if name == 'DONE':
                #     FORCE_STOP = True

                q.task_done()
                # camrea one
    finally:
        cv2.destroyAllWindows()

    t1.join()
    t2.join()

if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
