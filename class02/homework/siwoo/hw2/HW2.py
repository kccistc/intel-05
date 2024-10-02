#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np

FORCE_STOP = False

def thread_cam1(q):
    global FORCE_STOP
    cap = cv2.VideoCapture("resources/conveyor.mp4")  # 첫 번째 비디오 파일

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # "VIDEO:Cam1 live" 큐에 프레임 정보 추가
        q.put(("VIDEO:Cam1 live", frame))

        # 비율 계산 (임의 값으로 설정)
        x_ratio = np.random.random() * 100
        circle_ratio = np.random.random() * 100

        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

    cap.release()
    q.put(('DONE', None))


def thread_cam2(q):
    global FORCE_STOP
    cap = cv2.VideoCapture("resources/conveyor.mp4")  # 두 번째 비디오 파일

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # "VIDEO:Cam2 live" 큐에 프레임 정보 추가
        q.put(("VIDEO:Cam2 live", frame))

        # 색상 감지 비율 (임의 값으로 설정)
        ratio = np.random.random() * 100

        print(f"Color Detection Ratio: {ratio:.2f}%")

    cap.release()
    q.put(('DONE', None))


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

    # thread_cam1 및 thread_cam2 스레드 생성
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    while not FORCE_STOP:
        if cv2.waitKey(10) & 0xff == ord('q'):
            FORCE_STOP = True
            break

        try:
            # 큐에서 항목을 가져옴
            name, data = q.get(timeout=1)
            
            # 비디오 출력
            if name == "VIDEO:Cam1 live":
                imshow("Cam1 live", data, pos=(0, 0))
            elif name == "VIDEO:Cam2 live":
                imshow("Cam2 live", data, pos=(640, 0))

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

        except Empty:
            pass

    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
