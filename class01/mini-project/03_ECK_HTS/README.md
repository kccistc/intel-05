
# 개요 및 소개

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Dongle&size=60&pause=1000&color=F249F7&background=FFFFFF8D&repeat=false&random=false&width=800&lines=Hand+gesture+classification,+%EC%86%90%EC%9C%BC%EB%A1%9C+%EB%A7%90%ED%95%B4%EC%9A%94)](https://git.io/typing-svg)

<br>   

# 팀원역활     

<br>   

- #### 김은찬: 손 제스처 분류 모델 제작 검토&확인, ppt 제작&발표

- #### 한태섭: 손 제스처 분류 모델 탐색 및 평가,간단한 이벤트 핸들링, ppt 제작&발표

     
<br>  

   
<br>   
   
# 프로젝트 문제 정의
“해외여행객 247% 급증… 해외여행 열풍 올해도 이어지나”
“폭염 속 ‘12시간 근무’ 골프장 캐디…. ‘기절.화상이 일상’ ”
"대형마트 장애인용 쇼핑카트 의무화… 당사자에겐 유명무실” 

- #### 현대 사회에서 개인의 짐을 옮길 상황이 증가하고 있으며, 이러한 상황에서 물체를 안전하고 효율적으로 운반함으로써 고객에게 편리한 경험을 제공
   
<br>   

# 프로젝트 목표

1. 사람의 신체중 손을 인식하여, 특정 제스처를 분류하는 모델 생성

2. 특정 제스처에 따라 이벤트를 실행하는 기능 구현
    
   
<br>   

   
# 시스템 디자인 구상도
![Untitled Diagram drawio](https://github.com/kccistc/intel-04/assets/165994180/50373d35-aadc-4579-9411-e2de5274a67a)

- ##### 추후 보강
   
<br>   

#### 세부 시스템 개발 구상도

- ##### 준비중


<br>   
   
# 시스템 기술 구성도
- #### 손의 관절을 벡터값으로 변환하는 모델
- #### 그 값을 입력으로 받아 제스쳐 분류 모델
- #### 분류에 따라 정해진 기능을 하는 이벤트
   
<br>   
   
# 개발 진행
   
<br>   
   
### 1. 로보플로우의 오픈 데이터셋을 사용하여 제스처 인식 모델 생성


<br>   
   
### 2. 주먹 가위 보 세가지 class로 직접찍은 사진을 분류한 후 이미지 특징에 따라 제스처 분류 모델 생성

   
<br>   
   
### 3. pre training 된 제스처 인식 모델 사용

   
<br>   
   
### 4. 특정 제스처 몇초이상 인식시 이벤트가 발생하는 코드 추가

   
<br>   
   

# 프로젝트 시연 결과


<br>   

- ### 로보플로우의 오픈 데이터셋을 사용한 모델
![스크린샷 2024-09-27 13-10-43](https://github.com/user-attachments/assets/b1f2c4ea-d059-477f-b82d-8dd869e93bb0)


- ### 직접찍은 사진으로 학습시킨 모델
![스크린샷 2024-09-27 13-11-30](https://github.com/user-attachments/assets/1980760e-bc46-4b1f-ab34-7331cc6ecbf8)




- ### 미디어 파이프 사용
![스크린샷 2024-09-27 09-53-59](https://github.com/user-attachments/assets/75f8b993-f06a-444b-9a69-712995e3b4f0)




- ### 이벤트 처리 영상
![GIFMaker_me (1)](https://github.com/user-attachments/assets/8970557b-44c6-43cf-83dc-f4b6ee1c4612)

<br>   


# 시연 결과 분석
   1. 처음 인식한 손의 동작만 끝까지 tracking 하는 성질 활용
   2. 하드웨어 리소스 계산 필요 for edge
   3. 돌발 상황에 대한 예외처리 필요
     
<br>   

# 프로젝트 성과 및 향후 발전 방향
- ### 프로젝트 성과 
1. 사람의 손만 인식하여 제스처 분류 모델 발견
2. 추후 프로젝트에 이벤트 처리를 넣어 활용 가능
     
<br>   

- ### 향후 발전 방향
1. 처음 인식한 손만 track 하는 것이 아닌 원하는 사용자의 손을 track 하도록 발전 

     
<br>   
 

