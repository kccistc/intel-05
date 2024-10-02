#include <Servo.h>

Servo myServo;  // 서보 모터 객체 생성
int motorPin1 = 4;  // DC 모터 핀 설정
int motorPin2 = 5;  // DC 모터 핀 설정
int speed = 6;

void setup() {
  // 시리얼 통신 시작 (속도 9600bps)
  Serial.begin(9600);
  myServo.attach(9);
  // 모터 핀 설정 (필요 시 설정)
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(speed, OUTPUT);
}

void loop(){
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');  // Python으로부터 명령 수신
    
    // 서보 모터 왼쪽 회전 명령
    if (command == "left") {
      myServo.write(0);  // 서보 모터를 왼쪽으로 회전
      Serial.println("Turning left");
    } 
    // 서보 모터 오른쪽 회전 명령
    else if (command == "right") {
      myServo.write(180);  // 서보 모터를 오른쪽으로 회전
      Serial.println("Turning right");
    } 
    // DC 모터 정지 명령
    
    if (command == "stop") {
      digitalWrite(motorPin1, LOW);  // DC 모터 중지
      digitalWrite(motorPin2, LOW);  // DC 모터 중지
      analogWrite(speed, 0);  // 모터 속도 0 (정지)
      Serial.println("Motors stopped");
    }
    // DC 모터 천천히 회전 명령
    else if (command == "slowly") {
      digitalWrite(motorPin1, HIGH);  // DC 모터 한 방향으로 회전
      digitalWrite(motorPin2, LOW);
      analogWrite(speed, 50);  // 속도 50으로 천천히 회전
      Serial.println("Motors moving slowly");
    }
    else if (command == "go"){
      digitalWrite(motorPin1, HIGH);  // DC 모터 한 방향으로 회전
      digitalWrite(motorPin2, LOW);
      analogWrite(speed, 100);  // 속도 100으로 계속 회전
      Serial.println("Motors moving at normal speed");
  }
    }
}
