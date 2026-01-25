<div align="center">

# Demo



https://github.com/user-attachments/assets/efada602-d0fa-4c96-a055-8bef72c985bb



</div>

<br>
<br>

<div align="center">

# 🚗 Lane Segmentation for ADAS Stability

**차선 Segmentation 기반 주행 보조 시스템(LKA) 안정화 프로젝트**

<img src="readme_image.png">

<br>

👥 Team Members
<div align="center"> <table> <tr> <td align="center" width="200"> <a href="https://github.com/USERNAME1"> <img src="https://avatars.githubusercontent.com/u/52408669?v=4" width="120" style="border-radius:50%"/> <br> <strong>이호욱</strong> </a> <br> <sub>Project Lead</td>
<td align="center" width="200">
  <a href="https://github.com/Kwakjaemin1007">
    <img src="https://avatars.githubusercontent.com/u/100951256?v=4" width="120" style="border-radius:50%"/>
    <br>
    <strong>곽재민</strong>
  </a>
  <br>
  <sub>Modeling
</td>

<td align="center" width="200">
  <a href="https://github.com/lhjjsh8-sketch">
    <img src="https://avatars.githubusercontent.com/u/247216328?v=4" width="120" style="border-radius:50%"/>
    <br>
    <strong>임은석</strong>
  </a>
  <br>
  <sub>Post-processing
</td>

<td align="center" width="200">
  <a href="https://github.com/mnjjzi68-cmd">
    <img src="https://avatars.githubusercontent.com/u/251106564?v=4" width="120" style="border-radius:50%"/>
    <br>
    <strong>김민지</strong>
  </a>
  <br>
  <sub>Real-time Inference<br/>
</td>

</tr> </table> </div>

# 🏅 Tech Stack 🏅

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

</div>

<br>

## Project Overview

| 항목 | 내용 |
|:-----|:-----|
| **📅 Date** | 2026.01.16 ~ 2026.01.23 |
| **👥 Type** | 팀 프로젝트 (End-to-End CV) |
| **🎯 Goal** | 차선 단절 환경에서도 주행 가능한 차로 영역 안정화 |
| **🔧 Tech Stack** | PyTorch, UNet, YOLO-Seg, OpenCV, Kalman Filter |
| **📊 Dataset** | SDLane Dataset |

<br>

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [문제 정의](#-문제-정의)
- [데이터셋 구조](#-데이터셋-구조)
- [모델 아키텍처](#-모델-아키텍처)
- [후처리 전략](#-후처리-전략)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [실험 결과](#-실험-결과)
- [프로젝트 구조](#-프로젝트-구조)

<br>

---

## 🔍 프로젝트 소개

본 프로젝트는 **ADAS(Advanced Driver Assistance Systems)**,  
특히 **중앙차로유지장치(LKA)** 환경에서 발생하는 차선 인식 실패 문제를 해결하기 위한  
**Lane Segmentation 기반 인식 안정화 프로젝트**이다.

차선이 명확히 보이지 않는 교차로, 회전 구간, 악천후 환경에서도  
**주행 가능한 차로 영역을 얼마나 연속적으로 유지할 수 있는지**를 핵심 목표로 설정하였다.

---

## ❗ 문제 정의

### 발생하는 실제 문제
- 교차로 및 사거리에서 차선이 물리적으로 끊김
- 좌·우회전 구간에서 유도선 누락 또는 불완전
- 점선 차선, 강한 반사광, 야간 노이즈

### 기존 접근의 한계
- 단일 프레임 기반 segmentation 모델은  
  **차선이 완전히 사라진 구간에서 예측 근거 자체가 부족**
- 프레임 간 떨림(Jitter) 및 예측 소실 발생

---

## 📁 데이터셋 구조
```
SDLane/
├── images/
│ ├── train/
│ └── test/
├── labels/
│ ├── train/ # JSON (polyline)
│ └── test/
```


### 데이터 특징
- 실제 대한민국 도로 환경 기반
- 차선이 보이지 않는 구간에도 **가상 차선 annotation 포함**
- Polyline 형태의 중심선 레이블 제공

---

## 🏗️ 모델 아키텍처

### 1️⃣ UNet (Baseline)
- Encoder–Decoder 구조 기반 Semantic Segmentation
- 차선/배경 이진 분할
- 안정적인 기준 모델

### 2️⃣ UNet + GRU
- 연속 프레임 기반 Temporal Modeling
- 이전 프레임 정보를 활용해 예측 안정성 강화

### 3️⃣ YOLO Segmentation
- Single-stage Instance Segmentation
- 실시간 추론 가능 (ADAS 요구사항 반영)

---

## 🧠 후처리 전략

### 적용 기법
- **Morphological Closing**: 점선 차선 연결
- **Polynomial / Linear Regression**: 차선 연장
- **Ego-lane 중심 좌표 필터링**
- **State Freezing + Damping**: 오버슈팅 방지
- **Kalman Filter (w/ Velocity)**: 곡률 추종 안정화
- **BEV(Bird’s Eye View) 변환**: 곡률 계산 안정화

> 📌 모델의 예측 품질이 후처리 성능을 결정함을 실험적으로 확인

---

## 🔧 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd lane_segmentation_project
```
### 2. 환경 설치
```bash
pip install -r requirements.txt
```

### 🚀 사용 방법
### 📚 학습
```bash
PYTHONPATH=src python scripts/train.py \
    --data_path /dataset \
    --model unet \
    --epochs 60 \
    --batch_size 16 \
    --img_h 400 \
    --img_w 640
```

### 🎥 비디오 추론
```bash
python scripts/infer_video.py \
    --weights checkpoints/unet_best.onnx \
    --video sample.mp4 \
    --out output.mp4
```

### 📐 후처리 포함 추론
```bash
python scripts/infer_video_postprocess.py \
    --enable_poly \
    --thr 0.5
```
---
### 📊 실험 결과
### 평가 지표

- Dice Score
- IoU
- Precision / Recall/F1
- 정성 평가 (교차로, 회전 구간, 야간)

### 핵심 관찰

- 차선이 명확한 구간: 모든 모델 안정적
- 차선 단절 구간: 구조적 한계 명확히 드러남
- 후처리는 모델 예측 품질에 강하게 의존
---
### 📂 프로젝트 구조
```css
lane_segmentation_project/
│
├── src/
│   ├── segtool/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── models_factory.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── postprocess.py
│   │   └── utils.py
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── infer_video.py
│   └── infer_video_postprocess.py
│
├── configs/
├── checkpoints/
└── README.md
```
---


# 🧠 Key Insights & Lessons Learned
### 1️⃣ 차선 단절 문제는 인식 성능 이전의 문제였다

차선이 물리적으로 사라지는 교차로·횡단보도 구간에서는
모델의 정확도나 구조를 아무리 개선해도,
입력 정보 자체가 부족한 태스크 구조적 한계가 존재했다.

→ “더 좋은 모델”보다 “무엇을 예측해야 하는가”의 정의가 먼저라는 점을 명확히 인식하게 되었다.

### 2️⃣ 단일 프레임 기반 Segmentation은 명확한 한계를 가진다

UNet, YOLO Segmentation 모두
차선이 명확히 보이는 구간에서는 안정적인 성능을 보였지만,
차선이 가려지거나 사라지는 순간 예측이 즉시 붕괴되었다.

→ Lane Segmentation은 본질적으로 가시 정보 의존도가 매우 높은 태스크임을 확인했다.

### 3️⃣ Temporal Modeling은 보조 수단일 뿐, 근본 해법은 아니다

UNet + GRU를 통해 프레임 간 정보를 활용했지만,
차선 미검출 구간이 길어질수록 hidden state의 신뢰도 역시 급격히 저하되었다.

→ Temporal 정보는 안정성을 ‘조금’ 개선할 수는 있으나,
차선이 없는 상황을 복원하는 근본적인 해결책은 아니었다.

### 4️⃣ 후처리는 모델 성능에 강하게 종속된다

Kalman Filter, Regression, Morphological 연산 등
다양한 후처리 기법을 적용한 결과,

모델 예측이 안정적인 경우 → 후처리 효과 매우 우수

모델 예측이 불안정한 경우 → 후처리 결과 급격히 악화

→ 후처리는 마법이 아니라 증폭기이며,
잘못된 예측을 올바르게 만드는 도구는 아님을 확인했다.

### 5️⃣ “차선을 분할하는 것”과 “주행 가능한 차로를 유지하는 것”은 다른 문제다

프로젝트를 통해 가장 중요하게 얻은 인사이트는 다음과 같다.

Lane Segmentation 태스크와
Drivable Area / Ego-lane 유지 태스크는
문제 정의 자체가 다르다.

이는 향후

Lane topology 추론

HD Map / Navigation 정보 융합

Multi-sensor fusion
과 같은 접근이 필요한 이유를 명확히 보여준다.

### 6️⃣ 실패를 통해 현실적인 ADAS 방향성을 확인했다

본 프로젝트는 목표를 완전히 달성하지는 못했지만,

CV 모델이 실제 주행 환경에서 어디까지 유효한지

어디서 구조적 한계를 가지는지

어떤 정보가 추가로 필요해지는지

를 실험적으로 검증한 프로젝트였다.

→ 단순 성능 개선이 아닌,
현실적인 ADAS 문제 정의와 기술 선택의 중요성을 명확히 이해하게 되었다.
