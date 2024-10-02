
from argparse import ArgumentParser
from iotdemo import FactoryController

FORCE_STOP = False

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
                
        if key is '3':
            ctrl.red = False
            ctrl.green = True
            ctrl.orange = True
        
        if key is '4':
            ctrl.red = True
            ctrl.green = True
            ctrl.orange = False

        if key is '5':
            ctrl.red = True
            ctrl.green = False
            ctrl.orange = True
        
        if key is '6':
            ctrl.conveyor = False
        
        if key is '7':
            ctrl.push_actuator(0)
        
        if key is '8':
            ctrl.push_actuator(1)
           