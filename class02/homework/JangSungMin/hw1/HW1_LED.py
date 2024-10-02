from iotdemo import FactoryController
from argparse import ArgumentParser

# 들여쓰기 수정
parser = ArgumentParser(prog='python3 factory.py',
                        description="Factory tool")

# 인자 추가 방법 수정 (옵션 앞에 'd'를 제거하고 "--device"만 남김)
parser.add_argument("--device",
                    default=None,
                    type=str,
                    help="Arduino port")
args = parser.parse_args()

FORCE_STOP = False
with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:
        key = input()  # 사용자로부터 입력 받기
        inkey = int(key)

        # 'is' 대신 '==' 사용 (문자열 비교에서는 '=='을 사용해야 함)
        if key == '0':
            FORCE_STOP = True
        
        # 들여쓰기 수정
        if key == '3':
            ctrl.red = True
            ctrl.green = False
            ctrl.orange = False

        if key == '4':
            ctrl.red = False
            ctrl.green = False
            ctrl.orange = True
        
        if key == '5':
            ctrl.red = False
            ctrl.green = True
            ctrl.orange = False

        if key == '6':
            ctrl.red = False
            ctrl.green = True
            ctrl.orange = False
