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

FORCE_STOP=False
with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:

        key=input()
        inKey=int(key)
        if key is '0':
            FORCE_STOP=True
        if key is '1':
            ctrl.system_start()
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=True
        if key is '2':
            ctrl.system_stop()
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=True
        if key is '3':
            ctrl.red=False
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=True
        if key is '4':
            ctrl.red=True
            ctrl.orange=False
            ctrl.green=True
            ctrl.conveyor=True
        if key is '5':
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=False
            ctrl.conveyor=True
        if key is '6':
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=False
        if key is '7':
            ctrl.push_actuator(1)
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=True
        if key is '8':
            ctrl.push_actuator(1)
            ctrl.red=True
            ctrl.orange=True
            ctrl.green=True
            ctrl.conveyor=True

            