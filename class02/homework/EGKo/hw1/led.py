from argparse import ArgumentParser
from iotdemo import FactoryController

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
    
    ctrl.red = False
    ctrl.green = False
    ctrl.orange = False
    
    while not FORCE_STOP:
        
        key = input()
        inkey = int(key)
        
        if key == '0':
            ctrl.red = True
            ctrl.green = True
            ctrl.orange = True
            FORCE_STOP = True
        
        if key == '3':
            ctrl.red = False
            ctrl.green = True
            ctrl.orange = True