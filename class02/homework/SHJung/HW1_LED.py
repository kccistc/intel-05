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
        
        #종료
        if key is '0':
            FORCE_STOP = True
        
        #False가 켜지는거
        if key is '1':
            ctrl.red = False
            ctrl.orange = True
            ctrl.green = True
        
        if key is '2':
            ctrl.red = True
            ctrl.orange = False
            ctrl.green = True
            
        if key is '3':
            ctrl.red = True
            ctrl.orange = True
            ctrl.green = False
