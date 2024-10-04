#!/usr/bin/env python3

import os
import queue
import threading
from argparse import ArgumentParser
from queue import Queue
from time import sleep

import cv2
import numpy as np

FORCE_STOP = False


def thread_cam1(q: Queue[tuple[str, np.ndarray | None]]):
    global FORCE_STOP
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame information to queue for Cam1
        q.put(("CAM1: Conveyor Belt", frame))

        # Calculate random ratios for demonstration
        x_ratio = np.random.random() * 100
        circle_ratio = np.random.random() * 100

        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

    cap.release()
    q.put(("DONE", None))


def thread_cam2(q: Queue[tuple[str, np.ndarray | None]]):
    global FORCE_STOP
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame information to queue for Cam2
        q.put(("CAM2: Color Detection", frame))

        # Generate random ratio for color detection
        ratio = np.random.random() * 100

        print(f"Color Detection Ratio: {ratio:.2f}%")

    cap.release()
    q.put(("DONE", None))


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog="python3 factory.py", description="Factory tool")

    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    q = Queue()

    # Start threads for Cam1 and Cam2
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    try:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                FORCE_STOP = True
                break

            try:
                # Retrieve items from queue
                name, data = q.get(timeout=1)

                # Display video output
                if name == "CAM1: Conveyor Belt":
                    imshow("[Live] CAM1: Conveyor Belt", data, pos=(0, 0))
                elif name == "CAM2: Color Detection":
                    imshow("[Live] CAM2: Color Detection", data, pos=(640, 0))

                if name == "DONE":
                    FORCE_STOP = True

                q.task_done()

            except queue.Empty:
                pass

    finally:
        # Ensure to stop threads by setting FORCE_STOP before joining
        FORCE_STOP = True
        t1.join()
        t2.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
