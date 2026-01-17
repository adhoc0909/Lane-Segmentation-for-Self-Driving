# Lane-Segmentation-for-Self-Driving

# 🚗 ADAS Lane Segmentation
ADAS 실사용 환경에서 차선 인식 안정성을 분석·개선하기 위한 Lane Segmentation 프로젝트

---

## 🏅 Tech Stack 🏅
- Python
- PyTorch
- segmentation_models_pytorch
- OpenCV
- NumPy
- Albumentations
- Streamlit (Demo)

---

## 👥 Team
| 이름 | 역할 |
|---|---|
| 곽재민 | 팀원 |
| 이호욱 | 팀장 |
| 임은석 | 팀원 |
| 김민지 | 팀원 |

---

## 📌 Project Overview

| 항목 | 내용 |
|---|---|
| 📅 Date | YYYY.MM.DD ~ YYYY.MM.DD |
| 👥 Type | 팀 프로젝트 |
| 🎯 Goal | ADAS 환경에서 차선 인식 실패 구간을 대상으로 segmentation 기반 인식 안정성 분석 |
| 🔧 Tech Stack | PyTorch, SMP, Albumentations, OpenCV, Streamlit |
| 📊 Dataset | SDLand (42dot) |

---

## 📋 Table of Contents
- 프로젝트 소개
- 문제 정의
- 데이터셋
- 접근 방법
- 평가 전략
- 실험 결과
- 데모
- 프로젝트 구조
- 실행 방법

---

## 🎯 프로젝트 소개
본 프로젝트는 ADAS 실사용 환경에서 교차로, 회전 구간, 악천후 등으로 인해  
차선 인식 신뢰도가 저하되는 문제를 segmentation 관점에서 분석하는 것을 목표로 한다.

> (추후 내용 추가)

---

## ❓ 문제 정의
- ADAS 중앙차로 유지 기능이 비활성화되는 주요 상황 정리
- 차선 인식 실패 원인 가설 설정
- segmentation 접근의 필요성 정의

> (추후 내용 추가)

---

## 📊 데이터셋
- 데이터셋 개요
- 데이터 구성
- 라벨 구조
- EDA 요약
- 데이터 무결성 검증

> (추후 내용 추가)

---

## 🧠 접근 방법
- Baseline 모델 선정
- 모델 구조 설명
- 학습 전략
- 데이터 증강 전략

> (추후 내용 추가)

---

## 📈 평가 전략
- 정량 평가 지표 (IoU, Dice, Recall 등)
- 실패 구간 중심 평가 방식
- 정성 평가 방법

> (추후 내용 추가)

---

## 🧪 실험 결과
| Model | Backbone | Metric 1 | Metric 2 | Inference Time |
|---|---|---|---|---|
| TBD | TBD | - | - | - |

> (추후 업데이트)

---

## 🎬 데모
- 실시간 추론 데모
- Streamlit 기반 시각화
- 입력/출력 예시

> (추후 이미지/영상 추가)

---

## 🚀 실행 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-id/adas-lane-segmentation.git
cd adas-lane-segmentation
2. 환경 설정
bash
코드 복사
# dependency 설치
uv sync
3. 학습
bash
코드 복사
python src/train.py
4. 추론
bash
코드 복사
python src/infer.py
5. 데모 실행
bash
코드 복사
streamlit run streamlit_app/app.py
📁 프로젝트 구조
text
코드 복사
adas-lane-segmentation/
├── configs/
├── data/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── streamlit_app/
├── scripts/
├── outputs/
├── weights/
├── notebooks/
├── pyproject.toml
└── README.md
📝 License
This project is licensed under the MIT License.


# 🚗 SDLane Lane Segmentation Baseline (Ready-to-Experiment)

SDLane(42dot) 데이터셋으로 **차선 segmentation** baseline을 **학습/평가**까지 바로 수행할 수 있는 템플릿입니다.
이후에는 **config만 바꿔서 모델/백본/증강/하이퍼파라미터 실험**을 반복할 수 있도록 구성했습니다.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 데이터 경로 (train 폴더)
export SDLANE_ROOT=/path/to/SDLane/train

python scripts/train.py --config configs/default.yaml
python scripts/eval.py  --config configs/default.yaml
```

## Colab에서 실행할 경우
데이터셋 드라이브에 저장한 후

!git clone https://github.com/adhoc0909/Lane-Segmentation-for-Self-Driving.git
%cd Lane-Segmentation-for-Self-Driving/
%env SDLANE_ROOT=/path/to/SDLane/train
!pip install -r requirements.txt
!python scripts/train.py --config configs/default.yaml
!python scripts/eval.py  --config configs/default.yaml



## Config override 예시
```bash
python scripts/train.py --config configs/default.yaml   --paths.run_name exp002   --model.arch deeplabv3plus   --model.encoder resnet50   --train.batch_size 16   --train.epochs 50
```

## Structure
- `src/lane_seg/data`: split / dataset / transforms
- `src/lane_seg/models`: model factory / losses
- `src/lane_seg/engine`: train/val loops / checkpoint
- `src/lane_seg/evaluation`: metrics
- `scripts`: entrypoints (train/eval/infer)
