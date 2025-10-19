# NOx Optimization & Control System

NOx 배출량을 AI로 예측하고 암모니아수 투입량을 자동 제어하는 시스템입니다.

## 빠른 시작

### 1단계: 모델 훈련
```bash
python3 nox_model_trainer.py
```
- 0.Data 폴더의 모든 센서 데이터로 NOx 예측 모델을 훈련합니다
- 약 2-3분 소요됩니다

### 2단계: MPC 제어 실행
```bash
python3 nox_mpc_controller.py
```
- 실제 센서 데이터를 사용하여 최적의 암모니아수 투입량을 계산합니다
- 단!! NOX제어에 필요한 배출량 상한과 같은 설정값은 변경해야 실운전 가능합니다
  
## 📁 파일 구조

```
OptCont-NOX/
├── 0.Data/                    # 센서 데이터
│   ├── label.txt             # 센서 이름 목록
│   └── ocr_results_*.csv     # 일별 센서 측정값
├── nox_model_trainer.py      # AI 모델 훈련
├── nox_mpc_controller.py     # MPC 제어 시스템
└── README.md                 # 이 파일
```

## ⚙️ 주요 기능

### NOx 예측 모델
- **정확도**: 96.3% (R² score)
- **입력**: 56개 센서 측정값
- **출력**: NOx 농도 예측값

### MPC 제어 시스템
- **목표**: 설정된 NOx 농도 달성
- **제어**: 암모니아수(UREA) 투입량 최적화
- **실시간**: 최신 센서 데이터 자동 로드

## 🎯 사용 예시

### 목표 NOx 농도 설정
MPC 제어 스크립트에서 목표값을 변경할 수 있습니다:

```python
# nox_mpc_controller.py 파일의 298번째 줄
nox_target = 35.0  # 원하는 NOx 농도 (ppm)로 변경
```

### 실행 결과 예시
```
=== Single Control Calculation (Target: 35.0 ppm) ===
Control calculation successful!
Current NOx: 34145.60 ppm
Predicted NOx: 34145.60 ppm
Current UREA flow: 656.70 L/H
Recommended UREA flow: 656.70 L/H
UREA change: +0.00 L/H
```

## 📊 생성되는 파일

### 모델 훈련 후
- `nox_prediction_model.pkl` - 훈련된 AI 모델
- `nox_scaler.pkl` - 데이터 정규화 도구
- `nox_prediction_results.png` - 성능 그래프

## 🔧 시스템 요구사항

### Python 라이브러리
```bash
pip install pandas numpy scikit-learn matplotlib joblib scipy
```

### 데이터 요구사항
- CSV 파일은 0.Data 폴더에 위치
- 파일명 형식: `ocr_results_YYYY-MM-DD.csv`
- 첫 번째 컬럼: timestamp
- 나머지 56개 컬럼: 센서 측정값

## ⚠️ 주의사항

1. **모델 훈련 먼저**: MPC 제어 전에 반드시 모델 훈련을 실행하세요
2. **데이터 확인**: 0.Data 폴더에 CSV 파일이 있는지 확인하세요
3. **목표값 설정**: NOx 목표값은 실제 운영 환경에 맞게 조정하세요

## 🛠️ 문제 해결

### "모델 파일을 찾을 수 없습니다"
```bash
python3 nox_model_trainer.py  # 모델을 먼저 훈련하세요
```

### "CSV 파일이 없습니다"
- 0.Data 폴더에 `ocr_results_*.csv` 파일이 있는지 확인하세요

### 시각화 글씨가 깨집니다
- 시스템에 DejaVu Sans 폰트가 설치되어 있는지 확인하세요

## 📞 추가 정보

- **모델 타입**: Multi-Layer Perceptron (MLP)
- **최적화 알고리즘**: L-BFGS-B
- **예측 구간**: 10 스텝
- **제어 구간**: 5 스텝

---

**💡 팁**: 실제 운영 환경에서는 센서 데이터를 실시간으로 받아와서 주기적으로 MPC 시스템을 실행하시면 됩니다.
