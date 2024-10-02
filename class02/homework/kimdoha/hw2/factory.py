#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
# from openvino.inference_engine import IECore
from openvino.runtime import Core

from iotdemo import FactoryController

FORCE_STOP = False

def thread_cam1(q):
    # Initialize MotionDetector
    # motion_detector = MotionDetector(history=20, varThreshold=25)

    # Load and initialize OpenVINO
    # ie = IECore()
    # ie = Core()
    # model_xml = "models/abnormal_detection.xml"
    # model_bin = "models/abnormal_detection.bin"
    # net = ie.read_network(model=model_xml, weights=model_bin)
    # input_blob = next(iter(net.input_info))
    # output_blob = next(iter(net.outputs))
    # exec_net = ie.load_network(network=net, device_name="CPU")

    # Open video clip instead of camera device
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame))

        # Motion detect
        # motion_mask = motion_detector.apply(frame)

        # Enqueue "VIDEO:Cam1 detected", detected info
        # detected = cv2.bitwise_and(frame, frame, mask=motion_mask)
        # q.put(('VIDEO:Cam1 detected', detected.copy()))

        # Abnormal detect
        # frame_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
        # reshaped = frame_rgb.transpose((2, 0, 1))
        # preprocessed_numpy = ((reshaped / 255.0) - 0.5) * 2
        # batch_tensor = np.expand_dims(preprocessed_numpy, axis=0)

        # Inference OpenVINO
        # res = exec_net.infer(inputs={input_blob: batch_tensor})

        # Calculate ratios
        # output = res[output_blob][0]
        # x_ratio = output[0] * 100
        # circle_ratio = output[1] * 100
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # Enqueue for moving the actuator 1
        #  threshold_x = 50.0  # Set your threshold
        #  threshold_circle = 50.0  # Set your threshold
        #  if x_ratio > threshold_x or circle_ratio > threshold_circle:
        # q.put(('PUSH', 1))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # Initialize MotionDetector
    # motion_detector = MotionDetector(history=20, varThreshold=25)

    # Initialize ColorDetector
    # color_detector = ColorDetector(target_color='blue')

    # Open video clip
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))

        # Detect motion
        # motion_mask = motion_detector.apply(frame)

        # Enqueue "VIDEO:Cam2 detected", detected info
        # detected = cv2.bitwise_and(frame, frame, mask=motion_mask)
        # q.put(('VIDEO:Cam2 detected', detected.copy()))

        # # Detect color
        # color_mask = color_detector.detect(detected)

        # # Compute ratio
        # ratio = (np.sum(color_mask > 0) / color_mask.size) * 100
        # print(f"{color_detector.target_color.capitalize()} Ratio: {ratio:.2f}%")

        # # Enqueue to handle actuator 2
        # color_threshold = 10.0  # Set your threshold
        # if ratio > color_threshold:
        #     q.put(('PUSH', 2))

    cap.release()
    q.put(('DONE', None))
    exit()


# def imshow(title, frame, pos=None):
#     cv2.namedWindow(title, cv2.WINDOW_NORMAL)
#     if pos:
#         cv2.moveWindow(title, pos[0], pos[1])
#     cv2.imshow(title, frame)

def imshow(title, frame, pos=None):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하게 설정
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])  # pos 값이 None이 아닐 때 창 위치 이동
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

    # Create thread_cam1 and thread_cam2 threads and start them
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            # Get an item from the queue
            try:
                name, data = q.get(timeout=1)
            except Empty:
                continue

            # Show videos with titles 'Cam1 live' and 'Cam2 live'
            print(name[11:])
            if name[11:] == 'live':
                imshow(name, data)
            elif name[11:] == 'detected':
                imshow(name, data)

            # Control actuator if name == 'PUSH'
            if name == 'PUSH':
                actuator_number = data
                ctrl.move_actuator(actuator_number)

            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(1)


# #!/usr/bin/env python3

# import os
# import threading
# from argparse import ArgumentParser
# from queue import Empty, Queue
# from time import sleep

# import cv2
# import numpy as np
# from openvino.inference_engine import IECore

# from iotdemo import FactoryController

# FORCE_STOP = False


# def thread_cam1(q):
#     # TODO: MotionDetector

#     # TODO: Load and initialize OpenVINO

#     # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.

#     while not FORCE_STOP:
#         sleep(0.03)
#         _, frame = cap.read()
#         if frame is None:
#             break

#         # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info

#         # TODO: Motion detect

#         # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

#         # abnormal detect
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         reshaped = detected[:, :, [2, 1, 0]]
#         np_data = np.moveaxis(reshaped, -1, 0)
#         preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
#         batch_tensor = np.stack(preprocessed_numpy, axis=0)

#         # TODO: Inference OpenVINO

#         # TODO: Calculate ratios
#         print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

#         # TODO: in queue for moving the actuator 1

#     cap.release()
#     q.put(('DONE', None))
#     exit()


# def thread_cam2(q):
#     # TODO: MotionDetector

#     # TODO: ColorDetector

#     # TODO: HW2 Open "resources/conveyor.mp4" video clip

#     while not FORCE_STOP:
#         sleep(0.03)
#         _, frame = cap.read()
#         if frame is None:
#             break

#         # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info

#         # TODO: Detect motion

#         # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

#         # TODO: Detect color

#         # TODO: Compute ratio
#         print(f"{name}: {ratio:.2f}%")

#         # TODO: Enqueue to handle actuator 2

#     cap.release()
#     q.put(('DONE', None))
#     exit()


# def imshow(title, frame, pos=None):
#     cv2.namedWindow(title)
#     if pos:
#         cv2.moveWindow(title, pos[0], pos[1])
#     cv2.imshow(title, frame)


# def main():
#     global FORCE_STOP

#     parser = ArgumentParser(prog='python3 factory.py',
#                             description="Factory tool")

#     parser.add_argument("-d",
#                         "--device",
#                         default=None,
#                         type=str,
#                         help="Arduino port")
#     args = parser.parse_args()

#     # TODO: HW2 Create a Queue

#     # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

#     with FactoryController(args.device) as ctrl:
#         while not FORCE_STOP:
#             if cv2.waitKey(10) & 0xff == ord('q'):
#                 break

#             # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
#             # de-queue name and data

#             # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.

#             # TODO: Control actuator, name == 'PUSH'

#             if name == 'DONE':
#                 FORCE_STOP = True

#             q.task_done()

#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     try:
#         main()
#     except Exception:
#         os._exit()
