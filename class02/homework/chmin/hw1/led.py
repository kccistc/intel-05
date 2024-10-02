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
  
with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:
            
        key = input()
        inKey = int(key)
        
        if key is '0':
            FORCE_STOP = True
        if key is '3':
            ctrl.red = True
            ctrl.orange = False
            ctrl.green = False
