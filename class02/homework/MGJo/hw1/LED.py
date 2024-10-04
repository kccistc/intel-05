from iotdemo import FactoryController
from argparse import ArgumentParser

FORCE_STOP = False

parser = ArgumentParser(prog='python3 factory.py',
                        description="Factory tool")

parser.add_argument("-d", 
                    "--device",
                    default=None,
                    type=str,
                    help="Arduino port")
args = parser.parse_args()
with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:

        key = input()
        inKey = int(key)

        if key is '0':
            FORCE_STOP = True

        if key is '3':#red
            ctrl.red = False
            ctrl.orange = True
            ctrl.green = True
        
        if key is '4':#orange
            ctrl.red = True
            ctrl.orange = False
            ctrl.green = True
        
        if key is '5':#green
            ctrl.red = True
            ctrl.orange = True
            ctrl.green = False