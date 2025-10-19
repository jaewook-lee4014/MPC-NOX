#!/usr/bin/env python3
"""
NOx MPC (Model Predictive Control) 제어 시스템
- 저장된 NOx 예측 모델을 활용한 암모니아수 투입량 최적화
- 설정된 NOx 목표값 달성을 위한 UREA 유량 제어 추천
- 실시간 센서 데이터 기반 제어값 계산
"""

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class NOxMPCController:
    def __init__(self, model_path='nox_prediction_model.pkl', scaler_path='nox_scaler.pkl'):
        """
        MPC 컨트롤러 초기화
        
        Args:
            model_path: 훈련된 NOx 예측 모델 경로
            scaler_path: 데이터 스케일러 경로
        """
        self.model_data = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.model = self.model_data['model']
        self.selected_features = self.model_data['selected_features']
        self.target_col = self.model_data['target_col']
        
        # MPC 파라미터
        self.prediction_horizon = 10  # 예측 구간
        self.control_horizon = 5      # 제어 구간
        self.urea_flow_bounds = (0, 1000)  # UREA 유량 제약 (L/H)
        
        # 가중치
        self.weight_tracking = 1.0    # 목표값 추종 가중치
        self.weight_control = 0.1     # 제어 변화량 가중치
        
        print("NOx MPC Controller 초기화 완료")
        print(f"사용 특성 수: {len(self.selected_features)}")
        print(f"예측 구간: {self.prediction_horizon}, 제어 구간: {self.control_horizon}")
    
    def normalize_sensor_data(self, sensor_data):
        """
        센서 데이터 정규화
        
        Args:
            sensor_data: 현재 센서 측정값 딕셔너리
            
        Returns:
            정규화된 센서 데이터 배열
        """
        # 모델 특성에 맞는 데이터 구성
        feature_vector = []
        for feature in self.selected_features:
            if feature in sensor_data:
                feature_vector.append(sensor_data[feature])
            else:
                # 누락된 특성은 0으로 처리 (정규화된 평균값)
                feature_vector.append(0.5)
                print(f"경고: 특성 '{feature}'이 센서 데이터에 없습니다. 기본값 사용.")
        
        # 스케일러 적용 (역변환 후 재변환으로 정규화)
        feature_array = np.array(feature_vector).reshape(1, -1)
        return feature_array
    
    def predict_nox(self, sensor_data, urea_flow=None):
        """
        현재 센서 데이터와 UREA 유량으로 NOx 농도 예측
        
        Args:
            sensor_data: 센서 측정값 딕셔너리
            urea_flow: UREA 유량 (L/H), None이면 현재값 사용
            
        Returns:
            예측된 NOx 농도 (정규화된 값)
        """
        # UREA 유량 업데이트
        if urea_flow is not None:
            sensor_data = sensor_data.copy()
            sensor_data['15. UREA FLOW (L/H)'] = urea_flow
        
        # 데이터 정규화
        normalized_data = self.normalize_sensor_data(sensor_data)
        
        # NOx 예측
        nox_prediction = self.model.predict(normalized_data)[0]
        return nox_prediction
    
    def mpc_objective(self, urea_sequence, sensor_data, nox_target, current_urea):
        """
        MPC 목적함수: NOx 목표값 추종 + 제어 변화량 최소화
        
        Args:
            urea_sequence: 제어 구간의 UREA 유량 시퀀스
            sensor_data: 현재 센서 데이터
            nox_target: NOx 목표값 (정규화된 값)
            current_urea: 현재 UREA 유량
            
        Returns:
            목적함수 값
        """
        cost = 0.0
        prev_urea = current_urea
        
        for i in range(self.control_horizon):
            urea_flow = urea_sequence[i]
            
            # NOx 예측
            predicted_nox = self.predict_nox(sensor_data, urea_flow)
            
            # 목표값 추종 비용
            tracking_error = (predicted_nox - nox_target) ** 2
            cost += self.weight_tracking * tracking_error
            
            # 제어 변화량 비용 (급격한 변화 방지)
            control_change = (urea_flow - prev_urea) ** 2
            cost += self.weight_control * control_change
            
            prev_urea = urea_flow
            
            # 센서 데이터 업데이트 (단순화: UREA만 변경)
            sensor_data = sensor_data.copy()
            sensor_data['15. UREA FLOW (L/H)'] = urea_flow
        
        return cost
    
    def calculate_optimal_urea(self, sensor_data, nox_target_ppm, current_urea_flow=None):
        """
        최적 UREA 유량 계산
        
        Args:
            sensor_data: 현재 센서 측정값 딕셔너리
            nox_target_ppm: NOx 목표값 (ppm, 실제 단위)
            current_urea_flow: 현재 UREA 유량, None이면 센서 데이터에서 가져옴
            
        Returns:
            최적 UREA 유량 추천값 딕셔너리
        """
        # 현재 UREA 유량
        if current_urea_flow is None:
            current_urea_flow = sensor_data.get('15. UREA FLOW (L/H)', 0)
        
        # NOx 목표값 정규화 (간단한 추정: 0-100 ppm을 0-1로 매핑)
        nox_target_normalized = min(max(nox_target_ppm / 100.0, 0), 1)
        
        # 현재 NOx 농도 예측
        current_nox = self.predict_nox(sensor_data)
        
        # 제어 변수 초기값 (현재 유량 유지)
        initial_urea_sequence = np.full(self.control_horizon, current_urea_flow)
        
        # 제약 조건: UREA 유량 범위
        bounds = [self.urea_flow_bounds for _ in range(self.control_horizon)]
        
        # 최적화 실행
        result = minimize(
            fun=self.mpc_objective,
            x0=initial_urea_sequence,
            args=(sensor_data, nox_target_normalized, current_urea_flow),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            optimal_urea_flow = result.x[0]  # 첫 번째 제어 구간 값 사용
            predicted_nox_with_control = self.predict_nox(sensor_data, optimal_urea_flow)
            
            return {
                'success': True,
                'optimal_urea_flow': round(optimal_urea_flow, 2),
                'current_urea_flow': round(current_urea_flow, 2),
                'urea_change': round(optimal_urea_flow - current_urea_flow, 2),
                'current_nox_normalized': round(current_nox, 4),
                'predicted_nox_normalized': round(predicted_nox_with_control, 4),
                'current_nox_ppm': round(current_nox * 100, 2),  # 역정규화 추정
                'predicted_nox_ppm': round(predicted_nox_with_control * 100, 2),
                'nox_target_ppm': nox_target_ppm,
                'optimization_cost': round(result.fun, 6)
            }
        else:
            return {
                'success': False,
                'message': f"최적화 실패: {result.message}",
                'current_urea_flow': round(current_urea_flow, 2),
                'current_nox_normalized': round(current_nox, 4),
                'current_nox_ppm': round(current_nox * 100, 2)
            }
    
    def run_control_loop(self, sensor_data, nox_target_ppm, max_iterations=10):
        """
        제어 루프 실행 (시뮬레이션)
        
        Args:
            sensor_data: 초기 센서 데이터
            nox_target_ppm: NOx 목표값 (ppm)
            max_iterations: 최대 반복 횟수
            
        Returns:
            제어 이력 리스트
        """
        control_history = []
        current_sensor_data = sensor_data.copy()
        
        print(f"Control loop started - NOx target: {nox_target_ppm} ppm")
        print("-" * 80)
        
        for iteration in range(max_iterations):
            # 최적 제어값 계산
            control_result = self.calculate_optimal_urea(
                current_sensor_data, nox_target_ppm
            )
            
            if not control_result['success']:
                print(f"Iteration {iteration + 1}: Optimization failed - {control_result['message']}")
                break
            
            # 결과 출력
            print(f"Iteration {iteration + 1}:")
            print(f"  Current NOx: {control_result['current_nox_ppm']:.2f} ppm")
            print(f"  Predicted NOx: {control_result['predicted_nox_ppm']:.2f} ppm")
            print(f"  Current UREA: {control_result['current_urea_flow']:.2f} L/H")
            print(f"  Recommended UREA: {control_result['optimal_urea_flow']:.2f} L/H")
            print(f"  UREA change: {control_result['urea_change']:+.2f} L/H")
            
            control_history.append(control_result)
            
            # 목표값 근사 달성 확인
            if abs(control_result['predicted_nox_ppm'] - nox_target_ppm) < 1.0:
                print(f"  ✓ Target achieved! (Error: {abs(control_result['predicted_nox_ppm'] - nox_target_ppm):.2f} ppm)")
                break
            
            # 다음 반복을 위한 센서 데이터 업데이트
            current_sensor_data['15. UREA FLOW (L/H)'] = control_result['optimal_urea_flow']
            
        print("-" * 80)
        print("Control loop completed")
        return control_history

def load_latest_sensor_data(data_dir='./0.Data'):
    """실제 센서 데이터에서 최신 데이터 로드"""
    import pandas as pd
    import os
    
    # 레이블 로드
    label_path = os.path.join(data_dir, 'label.txt')
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    # 가장 최신 CSV 파일 찾기
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort(reverse=True)  # 날짜순 내림차순 정렬
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")
    
    # 최신 파일에서 마지막 레코드 로드
    latest_file = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(latest_file)
    
    # 컬럼명 설정
    if len(labels) == df.shape[1] - 1:
        df.columns = ['timestamp'] + labels
    
    # 마지막 레코드를 딕셔너리로 변환
    latest_record = df.iloc[-1].to_dict()
    
    print(f"Loaded latest data from: {csv_files[0]}")
    print(f"Timestamp: {latest_record['timestamp']}")
    
    return latest_record

def main():
    """메인 실행 함수"""
    try:
        # MPC 컨트롤러 초기화
        controller = NOxMPCController()
        
        # 실제 센서 데이터 로드
        sensor_data = load_latest_sensor_data()
        
        print("=== NOx MPC Control System with Real Data ===")
        print(f"Current sensor data (first 10 features):")
        count = 0
        for key, value in sensor_data.items():
            if count >= 10:
                break
            if key != 'timestamp':
                print(f"  {key}: {value}")
                count += 1
        print("  ...")
        print()
        
        # 단일 제어값 계산
        nox_target = 35.0  # 목표 NOx 농도 (ppm)
        print(f"=== Single Control Calculation (Target: {nox_target} ppm) ===")
        result = controller.calculate_optimal_urea(sensor_data, nox_target)
        
        if result['success']:
            print("Control calculation successful!")
            print(f"Current NOx: {result['current_nox_ppm']:.2f} ppm")
            print(f"Predicted NOx: {result['predicted_nox_ppm']:.2f} ppm")
            print(f"Current UREA flow: {result['current_urea_flow']:.2f} L/H")
            print(f"Recommended UREA flow: {result['optimal_urea_flow']:.2f} L/H")
            print(f"UREA change: {result['urea_change']:+.2f} L/H")
        else:
            print("Control calculation failed:", result['message'])
        
        print()
        
        # 제어 루프 시뮬레이션
        print("=== Control Loop Simulation ===")
        controller.run_control_loop(sensor_data, nox_target, max_iterations=5)
        
    except FileNotFoundError as e:
        print(f"Error: Cannot find required files. {e}")
        print("Please run nox_model_trainer.py first to train the model.")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()