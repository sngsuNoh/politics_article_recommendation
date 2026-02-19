# 텍스트 임베딩 기반으로 편향도 예측하는 baseline 모델 학습

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime



def train_baseline(text_emb_path='data/embeddings/text_embeddings.npy',
                   data_path='data/processed/all_articles_labeled.csv',
                   save_dir='data/models',
                   test_size=0.3,
                   val_size=0.5,
                   random_state=42):
    
    """
    Args:
        text_emb_path: 텍스트 임베딩 경로
        data_path: 레이블 데이터 경로
        save_dir: 모델 저장 디렉토리
        test_size: Train에서 나머지 분리 비율 (0.3 = Val 0.15 + Test 0.15)
        val_size: 나머지에서 Val/Test 분리 비율 (0.5)
        random_state: 재현성 시드
    
    Returns:
        model: 학습된 모델 (XGB)
        results: 성능 지표
    """
    
    print("="*80)
    print("Baseline (Text Only) Model Training")
    print("="*80)
    
    # 데이터 로드
    print("\nLoading data...")
    text_emb = np.load(text_emb_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    y = df['bias_initial'].values
    
    print(f"  - Text embeddings: {text_emb.shape}")
    print(f"  - Labels: {len(y)}")
    print(f"  - Bias range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  - Bias mean: {y.mean():.2f} +/- {y.std():.2f}")
    
    # Train/Val/Test Split
    print("\nSplitting data...")
    # Train vs (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        text_emb, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Val vs Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=val_size, 
        random_state=random_state
    )
    
    print(f"  - Train: {len(X_train)} samples ({len(X_train)/len(text_emb)*100:.1f}%)")
    print(f"  - Val:   {len(X_val)} samples ({len(X_val)/len(text_emb)*100:.1f}%)")
    print(f"  - Test:  {len(X_test)} samples ({len(X_test)/len(text_emb)*100:.1f}%)")
    
    # 모델 학습
    print("\nTraining XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,      # 트리 개수
        max_depth=6,           # 트리 깊이
        learning_rate=0.1,     # 학습률
        subsample=0.8,         # 샘플링 비율
        colsample_bytree=0.8,  # Feature 샘플링 비율
        random_state=random_state,
        n_jobs=-1,             # 모든 CPU 코어 사용
        verbosity=1            # 학습 과정 출력
    )
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  Training completed in {train_time:.1f} seconds")
    
    # 평가 (Val)
    print("\nEvaluating on Validation Set...")
    y_val_pred = model.predict(X_val)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  - MAE:  {val_mae:.2f}")
    print(f"  - RMSE: {val_rmse:.2f}")
    print(f"  - R²:   {val_r2:.3f}")
    
    if val_mae < 25:
        print(f"  Target achieved! (MAE < 25)")
    else:
        print(f"  Target not met (MAE >= 25)")
    
    # 평가 (Test)
    print("\nEvaluating on Test Set...")
    y_test_pred = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  - MAE:  {test_mae:.2f}")
    print(f"  - RMSE: {test_rmse:.2f}")
    print(f"  - R²:   {test_r2:.3f}")
    
    # 언론사별 성능 분석
    print("\nPerformance by Media (Top 5 & Bottom 5)...")
    
    # Test 인덱스 추출
    indices = np.arange(len(text_emb))
    train_idx, temp_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, random_state=random_state)
    
    test_df = df.iloc[test_idx].copy()
    test_df['prediction'] = y_test_pred
    test_df['error'] = np.abs(y_test_pred - y_test)
    
    media_mae = test_df.groupby('언론사')['error'].mean().sort_values()
    
    print("\n  Best Predictions (Lowest MAE):")
    for media, mae in media_mae.head(5).items():
        print(f"    {media}: MAE = {mae:.2f}")
    
    print("\n  Worst Predictions (Highest MAE):")
    for media, mae in media_mae.tail(5).items():
        print(f"    {media}: MAE = {mae:.2f}")
    
    # 저장
    print("\nSaving model...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 저장
    model_path = os.path.join(save_dir, 'bias_model_baseline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_path}")
    
    # 결과 저장
    results = {
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'train_time': train_time,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(save_dir, 'baseline_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Results saved: {results_path}")
    
    # 결과 요약
    print("\n" + "="*80)
    print("BASELINE MODEL SUMMARY")
    print("="*80)
    print(f"Model Type:       XGBoost Regressor")
    print(f"Input Features:   Text Embeddings (768-dim)")
    print(f"Training Samples: {len(X_train)}")
    print(f"Training Time:    {train_time:.1f} seconds")
    print(f"\nValidation MAE:   {val_mae:.2f}")
    print(f"Test MAE:         {test_mae:.2f}")
    print(f"Test R²:          {test_r2:.3f}")
    print("="*80)
    
    return model, results


def analyze_predictions(model, 
                       text_emb_path='data/embeddings/text_embeddings.npy',
                       data_path='data/processed/all_articles_labeled.csv',
                       n_samples=10):
    """
    예측 결과 샘플 분석
    
    Args:
        model: 학습된 모델
        text_emb_path: 임베딩 경로
        data_path: 데이터 경로
        n_samples: 분석할 샘플 수
    """
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS (Sample)")
    print("="*80)
    
    # 데이터 로드
    text_emb = np.load(text_emb_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 랜덤 샘플 선택
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), n_samples, replace=False)
    
    # 예측
    X_sample = text_emb[sample_indices]
    y_true = df.iloc[sample_indices]['bias_initial'].values
    y_pred = model.predict(X_sample)
    
    # 출력
    for i, idx in enumerate(sample_indices):
        article = df.iloc[idx]
        true_bias = y_true[i]
        pred_bias = y_pred[i]
        error = abs(pred_bias - true_bias)
        
        print(f"\n#{i+1} | Index: {idx}")
        print(f"  Title:      {article['제목'][:60]}...")
        print(f"  Media:      {article['언론사']}")
        print(f"  True Bias:  {true_bias:+.1f}")
        print(f"  Pred Bias:  {pred_bias:+.1f}")
        print(f"  Error:      {error:.1f}")
        
        if error < 10:
            print(f"  Status:     Excellent")
        elif error < 20:
            print(f"  Status:     Good")
        else:
            print(f"  Status:     Poor")


if __name__ == '__main__':
    # Baseline 모델 학습
    model, results = train_baseline()
    
    # 예측 샘플 분석
    analyze_predictions(model, n_samples=10)
    
    print("\nDone")