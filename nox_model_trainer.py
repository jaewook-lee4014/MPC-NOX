#!/usr/bin/env python3
"""
NOx 예측 모델 개발 및 저장 스크립트
- 다일간 데이터를 활용한 NOx 예측 모델 훈련
- 최적화된 특성 선택 및 전처리
- 모델 성능 평가 및 저장
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NOxModelTrainer:
    def __init__(self, data_dir='./0.Data'):
        self.data_dir = data_dir
        self.label_path = os.path.join(data_dir, 'label.txt')
        self.scaler = MinMaxScaler()
        self.model = None
        self.selected_features = None
        self.target_col = '43. NOx (ppm)'
        
    def load_labels(self):
        """레이블 파일 로드"""
        with open(self.label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    
    def load_all_data(self):
        """모든 CSV 파일을 로드하여 통합 데이터셋 생성"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        labels = self.load_labels()
        
        all_data = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            
            # 컬럼명 설정
            if len(labels) == df.shape[1] - 1:
                df.columns = ['timestamp'] + labels
            
            all_data.append(df)
            print(f"로드된 파일: {file}, 레코드 수: {len(df)}")
        
        # 모든 데이터 통합
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"총 통합 데이터 크기: {combined_df.shape}")
        return combined_df
    
    def preprocess_data(self, df):
        """데이터 전처리"""
        print("데이터 전처리 시작...")
        
        # 타임스탬프 처리
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 숫자형 컬럼 식별
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 분산이 0이거나 고유값이 적은 컬럼 제거
        drop_cols = []
        for col in numeric_cols:
            if df[col].var() == 0 or pd.isna(df[col].var()) or df[col].nunique() <= 2:
                drop_cols.append(col)
        
        if drop_cols:
            print(f"제거된 컬럼: {drop_cols}")
            df = df.drop(columns=drop_cols)
            numeric_cols = [col for col in numeric_cols if col not in drop_cols]
        
        # 결측치 처리
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # 정규화
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        print("데이터 전처리 완료")
        return df, numeric_cols
    
    def select_features(self, df, numeric_cols, correlation_threshold=0.2):
        """NOx와 상관관계 높은 특성 선택"""
        if self.target_col not in df.columns:
            raise ValueError(f"타겟 컬럼 '{self.target_col}'이 데이터에 없습니다.")
        
        # NOx와의 상관관계 계산
        corr_matrix = df[numeric_cols].corr()
        nox_correlations = corr_matrix[self.target_col].abs().sort_values(ascending=False)
        
        # 타겟 컬럼 제외하고 상관관계 높은 특성 선택
        self.selected_features = nox_correlations[nox_correlations >= correlation_threshold].index.tolist()
        if self.target_col in self.selected_features:
            self.selected_features.remove(self.target_col)
        
        print(f"선택된 특성 수: {len(self.selected_features)}")
        print("상위 10개 특성:")
        for i, feature in enumerate(self.selected_features[:10]):
            print(f"{i+1}. {feature}: {nox_correlations[feature]:.3f}")
        
        return self.selected_features
    
    def train_model(self, df, test_ratio=0.2):
        """모델 훈련"""
        print("모델 훈련 시작...")
        
        # 특성과 타겟 분리
        X = df[self.selected_features]
        y = df[self.target_col]
        
        # 시간순 분할 (최신 데이터를 테스트용으로)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"훈련 데이터: {len(X_train)}, 테스트 데이터: {len(X_test)}")
        
        # 하이퍼파라미터 튜닝
        param_grid = {
            'hidden_layer_sizes': [(64, 32, 16), (100, 50, 25), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [1000, 2000]
        }
        
        mlp = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"최적 파라미터: {grid_search.best_params_}")
        
        # 예측 및 평가
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 성능 지표 계산
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n=== 모델 성능 평가 ===")
        print(f"훈련 데이터 - MSE: {train_mse:.6f}, R²: {train_r2:.4f}, MAE: {train_mae:.6f}")
        print(f"테스트 데이터 - MSE: {test_mse:.6f}, R²: {test_r2:.4f}, MAE: {test_mae:.6f}")
        
        return {
            'train_mse': train_mse, 'train_r2': train_r2, 'train_mae': train_mae,
            'test_mse': test_mse, 'test_r2': test_r2, 'test_mae': test_mae,
            'X_test': X_test, 'y_test': y_test, 'y_test_pred': y_test_pred
        }
    
    def save_model(self, model_path='nox_prediction_model.pkl', scaler_path='nox_scaler.pkl'):
        """모델과 스케일러 저장"""
        if self.model is None:
            raise ValueError("훈련된 모델이 없습니다. 먼저 train_model()을 실행하세요.")
        
        # 모델 메타데이터 포함
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'target_col': self.target_col
        }
        
        joblib.dump(model_data, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"모델 저장 완료: {model_path}")
        print(f"스케일러 저장 완료: {scaler_path}")
    
    def plot_predictions(self, results, save_path='nox_prediction_results.png'):
        """예측 결과 시각화"""
        y_test = results['y_test']
        y_test_pred = results['y_test_pred']
        
        plt.figure(figsize=(15, 5))
        plt.rcParams['font.family'] = 'DejaVu Sans'  # English font
        
        # 시계열 예측 결과
        plt.subplot(1, 2, 1)
        sample_indices = range(0, len(y_test), max(1, len(y_test)//100))  # 샘플링
        plt.plot(sample_indices, y_test.iloc[sample_indices], 'b-', label='Actual', alpha=0.7)
        plt.plot(sample_indices, y_test_pred[sample_indices], 'r-', label='Predicted', alpha=0.7)
        plt.title('NOx Prediction Results (Time Series)')
        plt.xlabel('Sample Index')
        plt.ylabel('NOx (ppm) - Normalized')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 산점도
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual NOx (ppm)')
        plt.ylabel('Predicted NOx (ppm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved: {save_path}")

def main():
    """메인 실행 함수"""
    trainer = NOxModelTrainer()
    
    try:
        # 1. 데이터 로드
        df = trainer.load_all_data()
        
        # 2. 전처리
        df, numeric_cols = trainer.preprocess_data(df)
        
        # 3. 특성 선택
        selected_features = trainer.select_features(df, numeric_cols)
        
        # 4. 모델 훈련
        results = trainer.train_model(df)
        
        # 5. 모델 저장
        trainer.save_model()
        
        # 6. 결과 시각화
        trainer.plot_predictions(results)
        
        print("\n=== 모델 개발 완료 ===")
        print("다음 파일들이 생성되었습니다:")
        print("- nox_prediction_model.pkl (예측 모델)")
        print("- nox_scaler.pkl (데이터 스케일러)")
        print("- nox_prediction_results.png (성능 시각화)")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()