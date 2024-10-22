# 상공회의소 서울기술교육센터 인텔교육 5기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-05.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 4기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 뭔가 센스있는 팀명
<프로젝트 요약>
* Members
  | Name | Role |
  |----|----|
  | 채치수 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 송태섭 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 정대만 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 채소연 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 권준호 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
* Project Github : https://github.com/goodsense/project_awesome.git
* 발표자료 : https://github.com/goodsense/project_aewsome/doc/slide.ppt

### Team: 북텔5기
<스마트 AI 도서관 (관리) 시스템>
  - 목적: 사서 및 관리자 업무 보조 시스템 (+도서관 이용자 편의 지향)
  - 문제: 최근 사서직 구인난 환경 속에서
    - 1) 사서직의 과중된 업무량
    - 2) 현행 업무환경의 제약
  - 목표: 문제 개선을 통해
    - 1) 사서직 업무 업무량 경감 및 효율 향상
    - 2) 도서관 이용자 UX 혁신
  - 방법: AI 기술 및 IT 기술 적용
* Members
  | Name | Role |
  |----|----|
  | 윤태준 | Project leading(선장),프로젝트를 총괄 및 AI Machine Learning |
  | 정태현 | Project managing(엔지니어), 모델/모듈 별 성능 분석 및 이력 관리, 이슈 공유. |
  | 최정호 | Back-end Building(갑판장), 프로젝트 토대가 되는 DataBase 구현, 사서 현직 인터뷰 내용 수집. |
  | 정상훈 | AI modeling(항해사), 타겟 모델 결과가 나오도록 AI model을 선택, data 수집, training 수행. |
* Project Github : https://github.com/yspsk1994/AI_Team_1.git
* 발표자료 : https://github.com/yspsk1994/AI_Team_1.git/doc/presentation.ppt

### Team: SignalMasters
<교통 제어 수신호 인식 시스템>
  - 목적: 교통 제어 수신호 인식 및 사고 예방
  - 문제: 수신호 도중 사망사고 예방 및 원활한 교통 환경 조성
    - 1) 자동차전용도로서 '차 고장 수신호' 하던 60대, SUV에 치여 사망 (24.03)
    - 2) '차량 고장' 수신호 하다 참변...뒷차에 치여 사망 (24.09)
  - 목표: 문제 개선을 통해
    - 1) 교통 제어 수신호 인식 모델 확립
    - 2) 차량 고장 도중 사망사고 예방
  - 방법: AI 기술 및 IT 기술 적용
* Members
  | Name | Role |
  |----|----|
  | 서창민 | Project leading(선장),프로젝트를 총괄 및 AI Machine Learning |
  | 박준수 | Project managing(엔지니어), 모델/모듈 별 성능 분석 및 이력 관리, 이슈 공유. |
  | 김도하 | Back-end Building(갑판장), 프로젝트 토대가 되는 DataBase 구현, 사서 현직 인터뷰 내용 수집. |
* Project Github : https://github.com/opmaksim/Signal-Project.git
* 발표자료 : https://github.com/opmaksim/Signal-Project/blob/main/doc/presentation.pptx

### Team: 가드로버(Guard Rover)
<자율주행 패트롤카>
_자율주행 로봇과 객체인식이 적용된 순찰 로봇_
  - 목적: 사람이 커버하기 힘든 공간을 사람을 대신하여 주어진 영역을 순찰하는 로봇
  - 문제: 인구가 줄어감에 따라 사람이 커버하기 힘든 넓은 영역에 대해서 순찰 및 감시를 보조할 수단이 필요함
  - 목표 
      - a. ~~라이다센서를 통해 주변을 인식하여 자율주행 가능한 로봇을 구현.~~
      - (수정) a-1. 초음파센서를 기반으로 전면의 장애물을 회피하며 공간을 주행하는 로봇 구현.
      - b. 로봇이 카메라를 통해 주행 중 영상을 획득.
      - c. 획득한 영상에서 무기를 소지한 사람을 인식하고 사용자에게 알림을 보낸다.
* Members
  | Name | Role |
  |----|----|
  | 권시우 | RC카 조립, 모터 구동부 작업, 객체인식 모델 트레이닝 |
  | 박정우 | RC카 조립, 무선 컨트롤러 작업, 무선 통신 구성 |
  | 조명근 | 트레이닝 모델 파이 적용 및 최적화 작업, 영상처리 최적화 |
  | 박인혁 | ~~ROS2 베이스의 Lidar Sensor & 자율주행 담당~~ <br> (조장) 시스템 설계, ROS 사용검토, 작업파일 병합 및 디버깅 |
* Project Github : https://github.com/FaceDown77/intel-class-PJT03.git
* 발표자료 : https://github.com/FaceDown77/intel-class-PJT03.git/doc/presentation.pptx

### Team: 쫒GO
<스마트 캐리어, 스마트 카트>
  - 목적: 사람의 제스처 인식 후, 사람의 뒤를 따르는 체이싱 카트
  - 문제: 개인의 짐을 옮길 상황이 증가
    - 1) 골프장 캐디 인력 구인 난등 직접 짐을 옮기는것에 대한 거부감 증가
    - 2) 신체적 장애인들을 위한 캐리어 서비스의 필요
  - 목표: 문제 개선을 통해
    - 1) 카트를 직접 끌지 않음으로써 두 손의 자유를 제공
    - 2) 손의 제스처를 통해 원격 제어 서비스 제공
  - 방법: AI 기술 및 IT 기술 적용
* Members
  | Name | Role |
  |----|----|
  | 김동현 | Project leading(선장),프로젝트를 총괄, harddware setting  |
  | 고의근 | Project managing(엔지니어), person detect & color classification Model 생성. |
  | 김은찬 | Back-end Building(갑판장), 모델/모듈 별 성능 분석 및 이력 관리, hardware setting. |
  | 한태섭 | AI modeling(항해사), hand gesture detection 모델 적용. |
* Project Github : https://github.com/KEG012/openvino-AI-project
* 발표자료 : https://github.com/KEG012/openvino-AI-project/blob/main/doc/team4.odp

### Team: 테일러스위프트
<스마트 캐리어>
  - 목적: 사용자를 인식 후, 사용자를 따라다니는 캐리어
  - 문제: 배달이 안되는 특수한 상황에서 몸이 불편한 사람들이 많은 짐을 옮길 도구가 필요하다 생각함.
  - 목표
    - 1) 카트를 직접 끌지 않음으로써 두 손의 자유를 제공
    - 2) 웨어러블 기기를 사용하지 않고 사람을 추적함으로 사용자의 부담을 제거.
  - 방법: AI 기술 및 IT 기술 적용
* Members
  | Name | Role |
  |----|----|
  | 장성민 | Project leading(선장),프로젝트를 총괄, persondetect, 특징추출모델생성  |
  | 김지환 | Project managing(엔지니어), harddware setting.  |
  | 최재원 | Back-end Building(갑판장), 거리측정 프로그램 개발. |
  | 정진우 | AI modeling(항해사), harddware setting, 모델검수 . |
* Project Github : https://github.com/jsamez/05-tailer-swift
* 발표자료 : https://github.com/jsamez/05-tailer-swift/blob/main/SmartCarrier.pptx
