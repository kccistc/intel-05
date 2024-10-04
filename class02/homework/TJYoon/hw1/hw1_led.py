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
        intKey = int(key)
        
        if key is '0':
            FORCE_STOP = False
        
        if key is '3':
            print('push 3')
            ctrl.red = False
            ctrl.orange = False
            ctrl.green = False
            
        if key is '4':
            print('push 4')
            ctrl.red = True
            ctrl.orange = False
            ctrl.green = True
            
        if key is '5':
            print('push 5')
            ctrl.red = True
            ctrl.orange = True
            ctrl.green = False
        
