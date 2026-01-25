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
### 📌 프로젝트 요약

본 프로젝트는
차선 segmentation 기반 접근이 실제 ADAS 환경에서 어디까지 유효한지,
그리고 어디서 구조적 한계를 가지는지를 End-to-End로 검증한 프로젝트이다.

단순 성능 향상이 아닌,
문제 정의 → 실험 → 실패 → 인사이트 도출을 명확히 기록하는 데 목적을 둔다.
