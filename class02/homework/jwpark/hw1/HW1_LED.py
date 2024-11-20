from iotdemo import FactoryController
from argparse import ArgumentParser


parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

parser.add_argument("-d",
                    "--device",
                    default=None,
                    type=str,
                    help="Arduino port")
args = parser.parse_args()

FORCE_STOP = False

with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:

        key = input()
        inKey = int(key)

        if key is '1':
            FORCE_STOP = True

        if key is '2':
            ctrl.red = False
            ctrl.orange = True
            ctrl.green = False

        if key is '3':
            ctrl.red = True
            ctrl.orange = False
            ctrl.green = True