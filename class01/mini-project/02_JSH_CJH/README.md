### 0. 멤버

---

- 정상훈
- 최정호

### 1. 프로젝트 개요

---

- 다양한 한글 책 제목을 기존 OCR 모델에게 학습시켜 한글 인식 기능 향상 도모

### **2. 프로젝트 진행 환경**

---

- 환경 및 tools
  1. Ubuntu 22.04 (Linux 운영체제)
  2. VS Code tool
  3. Python language
- 모델 및 데이터 세트
  1. pre-trained paddle 모델
  2. 새로운 학습 데이터(한글 문장, 단어)
     - 문장: 4만여 개
     - 단어: 26만여 개
  https://github.com/user-attachments/assets/66d46d8b-6a8e-4e2a-b172-b0321fef9794

### 3. 프로젝트 취지/목표

---

- 취지
  - paddle OCR 모델의 한글 인식률 향상
- 근거
  - paddle OCR: Accuracy: **93% - 95%**
  - Google Cloud Vision OCR: Accuracy: **96% - 98%**
  - MicroSoft Azure OCR: Accuracy: **94% - 96%**

### 4. 시스템 학습 구성도

---

https://github.com/user-attachments/assets/efc437a9-a63a-4a94-a760-7a34341e2fc3

### 5. 시연 및 결과

---

두두두두두

### 6. 고찰 및 보완점

---

- **다양한 글꼴에 따른 인식률 저하**

https://github.com/user-attachments/assets/2c9040f0-9402-4598-9a4b-59e05414ecb0

- **데이터 선택 기준의 중요성**
  1. 한글 언어 특성상 학습 데이터 의존성 큼
  2. 특정 글꼴/형식 편향 지양
  3. 다양한 폰트 학습 필요
- **모델 셋팅 환경 중요성**
  1. 초기 설정 복잡:
     - 의존성에 의한 복잡한 설정 과정
       1. PaddleOCR는 PaddlePaddle이라는 특정 딥러닝 프레임워크에 의존
       2. PaddlePaddle의 버전, CUDA와 CuDNN 호환성 등 다양한 조건 맞춘 설치 요구
       3. 다양한 Python 라이브러리와 종속성이 필요
       4. 환경에 따라 라이브러리 버전 충돌이나 설치 오류가 발생
     - 사용자 직접 다운로드 및 설정
       1. 필요한 모델 파일을 직접 다운로드/설정
       2. 잘못된 모델을 다운로드/경로 설정 시 인식 과정에서 오류 발생
     - 환경 변수 및 경로 직접 설정:
       1. 데이터 경로, 모델 저장 위치, 출력 파일 경로 등을 사용자가 직접 지정
       2. 설정이 잘못될 경우 실행 중단
  2. 복잡한 명령어 옵션:
     - CLI(명령줄 인터페이스)를 통한 옵션이 복잡하고 많음
       1. 데이터 전처리 / 모델 학습 / 추론 설정을 각각 다르게 조정
  3. 많은 파라미터 튜닝:
     - 학습률(Learning Rate)
     - 배치 크기(Batch size)
     - 데이터 전처리 파라미터
     - 모델 아키텍처 선택
     - 하이퍼 파라미터 튜닝
