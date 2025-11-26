# FaceFeel: AI 기반 실시간 관상 분석 솔루션 (Face Physiognomy Analysis)

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Mesh-orange)
![Architecture](https://img.shields.io/badge/Architecture-OOP_%26_Modular-purple)

## 📖 프로젝트 개요 (Project Overview)
**FaceFeel**은 사용자의 얼굴 특징(눈매, 턱선, 얼굴형)을 실시간으로 정밀 분석하여 10가지 페르소나(동물상)로 분류하는 **AI 기반 엔지니어링 프로젝트**입니다.

단순한 스크립트 형태를 넘어, **객체 지향 프로그래밍(OOP)** 원칙을 적용하여 유지보수성과 확장성을 확보하였으며, **다수결 투표(Majority Voting)** 및 **완충 구간(Buffer Zone)** 알고리즘을 통해 실시간 데이터의 노이즈를 효과적으로 제어했습니다.

---

## 🛠 기술 스택 및 아키텍처 (Tech Stack & Architecture)

### 1. Core Technologies
* **Language:** Python 3.9+
* **Computer Vision:** OpenCV (cv2)
* **AI Model:** Google MediaPipe Face Mesh (468 3D Landmarks)
* **Data Structures:** Collections (Deque, Counter), Dataclasses

### 2. Engineering Features (구현 핵심)
* **OOP Refactoring:** `FaceAnalyzer`, `TextRenderer`, `FaceConfig` 클래스 분리를 통한 모듈화.
* **Type Hinting:** Python `typing` 모듈을 활용한 명시적 타입 정의로 코드 안정성 확보.
* **FPS Monitoring:** 실시간 성능 모니터링 기능 탑재.
* **State Management:** 분석 진행률(Progress) 및 결과 잠금(Lock) 상태 관리 로직 구현.

---

## 🚀 주요 기능 (Key Features)

### 1. 3단계 완충 분류 알고리즘 (3-Stage Buffer Logic)
* 경계값(Threshold) 부근의 데이터 떨림(Flickering) 현상을 방지하기 위해, 단순 이진 분류가 아닌 **[Low - Balanced - High]** 3단계 분류 체계를 도입했습니다.
* 이를 통해 '조화로운 밸런스형'과 같은 중간값을 확보하여 분석 결과의 안정성을 높였습니다.

### 2. 다수결 투표 시스템 (Majority Voting System)
* 단일 프레임(Single Frame) 분석의 오차를 줄이기 위해 시계열 데이터를 수집합니다.
* 수집된 표본 중 최빈값(Mode)을 최종 결과로 채택하여 신뢰도 95% 이상의 결과를 도출합니다.

### 3. 사용자 행동 제어 및 정규화 (Input Normalization)
* **Pose Estimation:** Yaw(좌우), Pitch(상하) 각도를 계산하여 정면 응시 유도.
* **Expression Filter:** 입 벌림 정도(Mouth Open Ratio)를 계산하여 무표정(Neutral) 데이터만 선별적으로 수집.

### 4. 시각화 및 UX (Visualization)
* **Progress Bar:** 데이터 수집 현황을 실시간 게이지로 시각화.
* **Korean Support:** Pillow 라이브러리를 활용한 한글 폰트 렌더링 엔진(`TextRenderer`) 자체 구현.

---

## 📂 프로젝트 구조 (Project Structure)

FaceFeel/
├── facefeel.py          # Main Application entry point
├── requirements.txt     # Dependencies list
├── README.md            # Project documentation
└── FaceFeel_Result.jpg  # (Auto-generated) Analysis result image

## 코드 모듈 상세
FaceConfig (@dataclass): 임계값, 버퍼 크기, 폰트 경로 등 설정값 중앙 관리.

FaceAnalyzer (Class): 얼굴 데이터 추출, 버퍼 관리, 투표 로직 등 핵심 엔진.

TextRenderer (Class): OpenCV 이미지와 PIL 간 변환을 통한 텍스트 렌더링 처리.

## 💻 실행 방법 (How to Run)
환경 설정 (Installation)

Bash

pip install -r requirements.txt
실행 (Run)

Bash

python facefeel.py

## 조작 방법 (Controls)

q: 프로그램 종료

r: 분석 초기화 (Reset)

s: 결과 이미지 저장 (Save)

## 🔍 관상 분류표 (Classification Matrix)

FaceFeel은 눈매(Eye Angle), 턱선(Jaw Shape), 얼굴형(Face Ratio)의 조합을 분석하여 총 10가지 페르소나로 분류합니다.

| 분류 (Type) | 눈매 (Eye) | 턱선 (Jaw) | 얼굴형 (Shape) | 특징 (Trait) |
| :--- | :--- | :--- | :--- | :--- |
| **냉철한 CEO (고양이상)** | 올라감 (Cat) | 강함 (Strong) | - | 카리스마, 리더십 |
| **시크한 모델 (뱀상)** | 올라감 (Cat) | 갸름함 (Soft) | - | 세련됨, 도회적 |
| **트렌디한 인싸 (여우상)** | 올라감 (Cat) | 보통 (Balanced) | - | 화려함, 매력 |
| **든든한 멘토 (대형견상)** | 처짐 (Dog) | 강함 (Strong) | - | 신뢰감, 우직함 |
| **순수한 힐러 (강아지상)** | 처짐 (Dog) | 갸름함 (Soft) | - | 보호본능, 순수 |
| **다정한 이웃 (사슴상)** | 처짐 (Dog) | 보통 (Balanced) | - | 편안함, 부드러움 |
| **열정적인 개척자 (늑대상)** | 보통 (Balanced) | 강함 (Strong) | - | 강인한 의지 |
| **지적인 학자 (토끼상)** | 보통 (Balanced) | 갸름함 (Soft) | 긴 얼굴 (Long) | 논리적, 차분함 |
| **행복한 쿼카 (다람쥐상)** | 보통 (Balanced) | 갸름함 (Soft) | 둥근 얼굴 (Round) | 긍정, 귀여움 |
| **천의 얼굴 (조화)** | 보통 (Balanced) | 보통 (Balanced) | - | 황금비율, 밸런스 |