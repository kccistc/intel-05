#!/usr/bin/env python3

import os
from argparse import ArgumentParser

from iotdemo import FactoryController

FORCE_STOP = False


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog="python3 factory.py", description="Factory tool")

    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    # TODO: HW2 Create a Queue

    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            key = input()
            if key == "0":
                FORCE_STOP = True

                ctrl.red = False
                ctrl.orange = False
                ctrl.green = False
            if key == "3":
                ctrl.red = False
                ctrl.orange = True
                ctrl.green = True

            if key == "4":
                ctrl.red = True
                ctrl.orange = False
                ctrl.green = True

            if key == "5":
                ctrl.red = True
                ctrl.orange = True
                ctrl.green = False
            print("WHy not..")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os._exit()
