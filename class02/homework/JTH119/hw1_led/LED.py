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
        if key is '0':
            print('00')
            FORCE_STOP = True

        if key is '3':
            print('33')
            ctrl.red = False
            ctrl.green = True
            ctrl.orange = True

        if key is '4':
            print('44')
            ctrl.red = True
            ctrl.green = False
            ctrl.orange = True

        if key is '5':
            print('55')
            ctrl.red = True
            ctrl.green = True
            ctrl.orange = False