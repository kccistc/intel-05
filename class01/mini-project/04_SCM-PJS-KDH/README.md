# OpenVINO 기반 포즈 인식 및 모터 제어 미니 프로젝트

## 팀원
- 서창민
- 박준수
- 김도하

## 프로젝트 배경
차량 고장 등 도로에서 불가피하게 차량을 움직일 수 없을 때 사람의 수신호를 인식하여 사고를 예방하기 위함
- ![image](https://github.com/user-attachments/assets/f9269aaf-fb24-4ef4-b2e1-c02d0b18750b)
- ![image](https://github.com/user-attachments/assets/71840e94-e016-48af-8e27-574afef61611)


## 프로젝트 개요
이 프로젝트는 **OpenVINO**를 활용하여 실시간 포즈 인식과 모터 제어를 구현하는 미니 프로젝트입니다.
사용자는 카메라를 통해 실시간으로 인식된 자세에 따라 서보 모터, 5V DC 모터를 제어할 수 있습니다.
이 시스템은 두 가지 주요 모델을 사용하여 동작합니다: **포즈 인식 모델**과 **모노 뎁스(mono-depth) 모델**입니다.

## 프로젝트 아키텍쳐
![미니프로젝트 아키텍쳐 drawio](https://github.com/user-attachments/assets/b22eb5b6-36b7-4f27-bbce-e5526387d7de)


## 사용된 모델
- **pose-estimation**: 사용자의 실시간 자세를 추적하여 주요 관절 위치를 감지합니다.
- **mono-depth**: 깊이 인식을 통해 사용자와 카메라 사이의 거리 정보를 추출합니다.

## 사용된 툴
- **Visual Studio Code**: Python 및 OpenVINO 모델 실행을 위한 IDE.
- **Arduino IDE**: 모터 제어용 C 코드를 작성 및 업로드하는 데 사용.

## 사용된 기술
- **OpenVINO**: Intel의 OpenVINO 툴킷을 사용하여 효율적인 딥러닝 모델을 추론하고 실시간 포즈 및 깊이 인식을 처리.
- **Python**: 포즈 인식 및 모노 뎁스 모델을 처리하고 Arduino와의 통신을 처리하는 스크립트 작성.
- **C**: Arduino에서 모터를 제어하는 코드 작성.

## 프로젝트 구조
```
project-root/
│
├── src/
│   ├── human-pose-estimation.py      # 포즈 인식 관련 코드
│   ├── vision-mono-depth.py           # 모노 뎁스 관련 코드
│   ├── motor_control.ino       # Arduino에서 모터 제어 코드
│
├── models/
│   ├── human-pose-estimation   # OpenVINO 포즈 인식 모델
│   └── MiDas_small        # OpenVINO 모노 뎁스 모델
│
├── README.md                   # 프로젝트 설명
└── requirements.txt            # Python 의존성 목록
```

## 실행 방법

### 1. 필수 소프트웨어 설치
- **VS Code** 및 **Arduino IDE** 설치
- Python 3.x 및 필요한 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```

### 2. OpenVINO 설정
- OpenVINO 환경을 설정하고, `models/` 폴더에 pose-estimation 및 mono-depth 모델을 다운로드합니다.

### 3. 코드 실행
- **포즈 인식 및 깊이 감지**를 위한 Python 코드 실행:
  ```bash
  python src/pose_estimation.py
  ```

### 4. Arduino에서 모터 제어 코드 업로드
- Arduino IDE를 열고, `src/motor_control.ino` 코드를 업로드합니다.

## 시연 영상

### Human Pose-Estimation 좌표 기반으로 네 가지 수신호 인식

1. Stop
- ![Stop Signal](images/screenshot_stop.png)

2. Slowly
- ![Slowly Signal](images/screenshot_slowly.png)

3. Go Left
- ![Go Left Signal](images/screenshot_go_left.png)

4. Go Right
- ![Go Right Signal](images/screenshot_go_right.png)

### 각 수신호에 따른 모터 제어

1. Stop
- ![Stop Motor](images/motor_stop.png)

2. Slowly
- ![Slowly Motor](images/motor_slowly.png)

3. Go Right, Left
- ![Turn Motor](images/motor_turn.png)
