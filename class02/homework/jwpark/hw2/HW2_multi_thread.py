#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
from openvino.runtime import Core  # Updated import for OpenVINO

from iotdemo import FactoryController

FORCE_STOP = False


def thread_cam1(q):
    # Load and initialize OpenVINO
    core = Core()  # Use Core instead of IECore
    model = core.read_model(model="resources/model.xml")
    compiled_model = core.compile_model(model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Preprocess frame for inference (resize to match model input)
        # frame_resized = cv2.resize(frame, (input_layer.shape[2], input_layer.shape[3]))
        # frame_resized = frame_resized.astype(np.float32)
        # frame_resized = frame_resized.transpose(2, 0, 1)  # HWC to CHW
        # frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

        # Inference (synchronous for now, but can be optimized with async if needed)
        # res = compiled_model([frame_resized])
        # res = res[output_layer]

        # Calculate ratios
        # x_ratio = res[0, 0] * 100
        # circle_ratio = res[0, 1] * 100
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # Enqueue for moving the actuator 1
        # if x_ratio > 50:  # 임계값 기준으로 제어
        #     q.put(("PUSH", 1))

    cap.release()
    q.put(('DONE', None))
    exit()



def thread_cam2(q):
    # MotionDetector
    #motion_detector = cv2.createBackgroundSubtractorMOG2()

    # ColorDetector
    def detect_color(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return mask

    # Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))

        # Detect motion
        # motion_mask = motion_detector.apply(frame)
        # contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 0:
        #     detected = frame.copy()
        #     for contour in contours:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         cv2.rectangle(detected, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #     q.put(("VIDEO:Cam2 detected", detected))

        # Detect color
        color_mask = detect_color(frame)
        color_ratio = np.sum(color_mask > 0) / (frame.shape[0] * frame.shape[1]) * 100

        # Compute ratio
        print(f"Cam2 Color Ratio: {color_ratio:.2f}%")

        # Enqueue to handle actuator 2
        # if color_ratio > 20:  # 임계값 기준
        #     q.put(("PUSH", 2))

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

    # Create a Queue
    q = Queue()

    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
                name, data = q.get(timeout=0.1)
            except Empty:
                continue

            # Show videos with titles of 'Cam1 live' and 'Cam2 live'
            #if name == "VIDEO:Cam1 live":
            imshow(name, data, pos=(0, 0))
            #elif name == "VIDEO:Cam2 live":
            # elif name == "VIDEO:Cam1 detected":
            #     imshow("Cam1 detected", data, pos=(0, 480))
            # elif name == "VIDEO:Cam2 detected":
            #     imshow("Cam2 detected", data, pos=(640, 480))

            # Control actuator, name == 'PUSH'
            # if name == 'PUSH':
            #     if data == 1:
            #         ctrl.push_actuator(1)
            #     elif data == 2:
            #         ctrl.push_actuator(2)

            # if name == 'DONE':
            #     FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
