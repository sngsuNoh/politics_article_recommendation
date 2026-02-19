# 텍스트 임베딩 + 그래프 임베딩으로 편향도 예측하는 full 모델 학습

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle
import os
from datetime import datetime


def train_full_model(text_emb_path='data/embeddings/text_embeddings.npy',
                     graph_emb_path='data/embeddings/graph_embeddings.npy',
                     graph_feat_path='data/embeddings/graph_features.csv',
                     data_path='data/processed/all_articles_labeled.csv',
                     baseline_model_path='data/models/bias_model_baseline.pkl',
                     save_dir='data/models',
                     test_size=0.3,
                     val_size=0.5,
                     random_state=42):
    """
    Args:
        text_emb_path: 텍스트 임베딩 경로 (479188, 768)
        graph_emb_path: 그래프 임베딩 경로 (479188, 64)
        graph_feat_path: 그래프 Feature 경로 (479188, 5)
        data_path: 레이블 데이터 경로
        baseline_model_path: Baseline 모델 경로 (비교용)
        save_dir: 모델 저장 디렉토리
        test_size: Train에서 나머지 분리 비율 (0.3 = Val 0.15 + Test 0.15)
        val_size: 나머지에서 Val/Test 분리 비율 (0.5)
        random_state: 재현성 시드

    Returns:
        model: 학습된 모델 (XGB)
        results: 성능 지표
    """

    print("="*80)
    print("Full Model (Text + Graph) Training")
    print("="*80)

    # 데이터 로드
    print("\nLoading data...")
    text_emb = np.load(text_emb_path)
    graph_emb = np.load(graph_emb_path)
    graph_feat = pd.read_csv(graph_feat_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    y = df['bias_initial'].values

    print(f"  - Text embeddings:  {text_emb.shape}")
    print(f"  - Graph embeddings: {graph_emb.shape}")
    print(f"  - Graph features:   {graph_feat.shape}")
    print(f"  - Labels: {len(y)}")
    print(f"  - Bias range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  - Bias mean: {y.mean():.2f} +/- {y.std():.2f}")

    # 크기 검증
    assert len(text_emb) == len(graph_emb) == len(graph_feat) == len(y), (
        f"Size mismatch: text={len(text_emb)}, graph_emb={len(graph_emb)}, "
        f"graph_feat={len(graph_feat)}, labels={len(y)}"
    )
    print(f"  Data consistency check passed ({len(y):,} samples)")

    # Baseline 모델 로드 (비교용)
    baseline_model = None
    if os.path.exists(baseline_model_path):
        print(f"\nLoading baseline model for comparison...")
        with open(baseline_model_path, 'rb') as f:
            baseline_model = pickle.load(f)
        print(f"  Baseline model loaded: {baseline_model_path}")
    else:
        print(f"\nBaseline model not found at {baseline_model_path}")
        print(f"  Comparison will be skipped.")

    # Feature 결합 (텍스트 + 그래프 임베딩 + 그래프 Feature)
    print("\nCombining features...")
    X = np.hstack([text_emb, graph_emb, graph_feat.values])
    text_dim = text_emb.shape[1]         # 768
    graph_emb_dim = graph_emb.shape[1]   # 64
    graph_feat_dim = graph_feat.shape[1] # 5
    total_dim = text_dim + graph_emb_dim + graph_feat_dim  # 837

    print(f"  - Text dim:        {text_dim}")
    print(f"  - Graph emb dim:   {graph_emb_dim}")
    print(f"  - Graph feat dim:  {graph_feat_dim}")
    print(f"  - Combined shape:  {X.shape} ({text_dim} + {graph_emb_dim} + {graph_feat_dim} = {total_dim})")

    # 그래프 Feature 통계
    print("\nGraph Feature Statistics:")
    for col in graph_feat.columns:
        values = graph_feat[col].values
        print(f"  - {col:20s}: mean={values.mean():.3f}, std={values.std():.3f}, "
              f"min={values.min():.3f}, max={values.max():.3f}")

    # Train/Val/Test Split (Baseline과 동일한 random_state로 동일한 split 보장)
    print("\nSplitting data (same split as baseline)...")

    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, random_state=random_state)

    X_train = X[train_idx]
    X_val   = X[val_idx]
    X_test  = X[test_idx]

    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    print(f"  - Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  - Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Baseline 성능 측정 (동일한 test set에서)
    baseline_test_mae = None
    if baseline_model is not None:
        print("\nBaseline performance on same test set...")
        X_test_text = text_emb[test_idx]
        y_test_pred_baseline = baseline_model.predict(X_test_text)
        baseline_test_mae  = mean_absolute_error(y_test, y_test_pred_baseline)
        baseline_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_baseline))
        baseline_test_r2   = r2_score(y_test, y_test_pred_baseline)

        print(f"  - Baseline MAE:  {baseline_test_mae:.2f}")
        print(f"  - Baseline RMSE: {baseline_test_rmse:.2f}")
        print(f"  - Baseline R²:   {baseline_test_r2:.3f}")

    # 모델 학습
    print("\nTraining XGBoost model (Enhanced)...")
    model = XGBRegressor(
        n_estimators=200,       # 트리 개수 (Baseline보다 증가)
        max_depth=8,            # 트리 깊이 (Baseline보다 증가)
        learning_rate=0.05,     # 학습률 (Baseline보다 감소)
        subsample=0.8,          # 샘플링 비율
        colsample_bytree=0.8,   # Feature 샘플링 비율
        min_child_weight=3,     # 리프 노드 최소 샘플 수
        gamma=0.1,              # 분기 최소 손실 감소
        random_state=random_state,
        n_jobs=-1,              # 모든 CPU 코어 사용
        verbosity=1,            # 학습 과정 출력
        early_stopping_rounds=20,  # 검증 손실 개선 없으면 조기 종료
    )

    start_time = datetime.now()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    train_time = (datetime.now() - start_time).total_seconds()

    print(f"  Training completed in {train_time:.1f} seconds")
    print(f"  Best iteration: {model.best_iteration}")

    # 평가 (Val)
    print("\nEvaluating on Validation Set...")
    y_val_pred = model.predict(X_val)

    val_mae  = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2   = r2_score(y_val, y_val_pred)

    print(f"  - MAE:  {val_mae:.2f}")
    print(f"  - RMSE: {val_rmse:.2f}")
    print(f"  - R²:   {val_r2:.3f}")

    if val_mae < 20:
        print(f"  Target achieved! (MAE < 20)")
    else:
        print(f"  Target not met (MAE >= 20)")

    # 평가 (Test)
    print("\nEvaluating on Test Set...")
    y_test_pred = model.predict(X_test)

    test_mae  = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2   = r2_score(y_test, y_test_pred)

    print(f"  - MAE:  {test_mae:.2f}")
    print(f"  - RMSE: {test_rmse:.2f}")
    print(f"  - R²:   {test_r2:.3f}")

    # Baseline 대비 개선량
    if baseline_test_mae is not None:
        improvement = baseline_test_mae - test_mae
        improvement_pct = (improvement / baseline_test_mae) * 100
        print(f"\n  Improvement over Baseline:")
        print(f"  - Baseline MAE: {baseline_test_mae:.2f}")
        print(f"  - Full MAE:     {test_mae:.2f}")
        print(f"  - Reduction:    {improvement:.2f} ({improvement_pct:.1f}%)")
    else:
        improvement = None
        improvement_pct = None

    # Feature Importance 분석
    print("\nFeature Importance Analysis...")
    importance = model.feature_importances_

    # 구간별 중요도 합산
    text_importance  = importance[:text_dim]
    graph_emb_importance  = importance[text_dim:text_dim + graph_emb_dim]
    graph_feat_importance = importance[text_dim + graph_emb_dim:]

    text_total      = text_importance.sum()
    graph_emb_total = graph_emb_importance.sum()
    graph_feat_total = graph_feat_importance.sum()
    total = text_total + graph_emb_total + graph_feat_total

    print(f"  - Text contribution:       {text_total:.3f} ({text_total/total*100:.1f}%)")
    print(f"  - Graph emb contribution:  {graph_emb_total:.3f} ({graph_emb_total/total*100:.1f}%)")
    print(f"  - Graph feat contribution: {graph_feat_total:.3f} ({graph_feat_total/total*100:.1f}%)")

    print(f"\n  Graph Feature Importance (individual):")
    for col, imp in zip(graph_feat.columns, graph_feat_importance):
        print(f"    - {col:20s}: {imp:.6f} ({imp/total*100:.2f}%)")

    most_important_graph_feat = graph_feat.columns[np.argmax(graph_feat_importance)]
    print(f"\n  Most important graph feature: {most_important_graph_feat}")

    # 언론사별 성능 분석
    print("\nPerformance by Media (Top 5 & Bottom 5)...")

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
    model_path = os.path.join(save_dir, 'bias_model_full.pkl')
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
        'best_iteration': int(model.best_iteration),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'input_dim': total_dim,
        'text_dim': text_dim,
        'graph_emb_dim': graph_emb_dim,
        'graph_feat_dim': graph_feat_dim,
        'baseline_test_mae': baseline_test_mae,
        'improvement_mae': improvement,
        'improvement_pct': improvement_pct,
        'text_contribution': float(text_total),
        'graph_emb_contribution': float(graph_emb_total),
        'graph_feat_contribution': float(graph_feat_total),
        'graph_feature_importance': {
            col: float(imp)
            for col, imp in zip(graph_feat.columns, graph_feat_importance)
        },
        'most_important_graph_feature': most_important_graph_feat,
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(save_dir, 'full_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Results saved: {results_path}")

    # 결과 요약
    print("\n" + "="*80)
    print("FULL MODEL SUMMARY")
    print("="*80)
    print(f"Model Type:       XGBoost Regressor (Enhanced)")
    print(f"Input Features:   Text ({text_dim}) + Graph emb ({graph_emb_dim}) + Graph feat ({graph_feat_dim}) = {total_dim}-dim")
    print(f"Training Samples: {len(X_train)}")
    print(f"Training Time:    {train_time:.1f} seconds")
    print(f"Best Iteration:   {model.best_iteration}")
    print(f"\nValidation MAE:   {val_mae:.2f}")
    print(f"Test MAE:         {test_mae:.2f}")
    print(f"Test R²:          {test_r2:.3f}")

    if baseline_test_mae is not None:
        print(f"\nImprovement over Baseline:")
        print(f"  Baseline MAE:   {baseline_test_mae:.2f}")
        print(f"  MAE Reduction:  {improvement:.2f} ({improvement_pct:.1f}%)")

    print(f"\nFeature Contribution:")
    print(f"  Text:       {text_total:.3f} ({text_total/total*100:.1f}%)")
    print(f"  Graph emb:  {graph_emb_total:.3f} ({graph_emb_total/total*100:.1f}%)")
    print(f"  Graph feat: {graph_feat_total:.3f} ({graph_feat_total/total*100:.1f}%)")
    print(f"  Most Important Graph Feature: {most_important_graph_feat}")
    print("="*80)

    return model, results


def analyze_graph_contribution(model,
                                text_emb_path='data/embeddings/text_embeddings.npy',
                                graph_emb_path='data/embeddings/graph_embeddings.npy',
                                graph_feat_path='data/embeddings/graph_features.csv',
                                data_path='data/processed/all_articles_labeled.csv',
                                baseline_model_path='data/models/bias_model_baseline.pkl',
                                n_samples=100):
    """
    Baseline 모델과 Full 모델을 동일한 샘플에 대해 비교하여
    그래프 Feature 및 임베딩이 실제로 예측을 얼마나 개선하는지 측정

    Args:
        model: 학습된 Full 모델
        text_emb_path: 텍스트 임베딩 경로
        graph_emb_path: 그래프 임베딩 경로
        graph_feat_path: 그래프 Feature 경로
        data_path: 레이블 데이터 경로
        baseline_model_path: Baseline 모델 경로
        n_samples: 분석할 샘플 수
    """
    print("\n" + "="*80)
    print("GRAPH CONTRIBUTION ANALYSIS")
    print("="*80)

    # 데이터 로드
    text_emb  = np.load(text_emb_path)
    graph_emb = np.load(graph_emb_path)
    graph_feat = pd.read_csv(graph_feat_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # Baseline 모델 로드
    print("\nLoading baseline model...")
    if not os.path.exists(baseline_model_path):
        print(f"  Baseline model not found: {baseline_model_path}")
        return

    with open(baseline_model_path, 'rb') as f:
        baseline_model = pickle.load(f)
    print(f"  Baseline model loaded")

    # 랜덤 샘플 선택
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), n_samples, replace=False)

    # 예측 비교
    print(f"\nComparing predictions on {n_samples} samples...")

    # 텍스트만 (Baseline)
    X_text_only = text_emb[sample_indices]
    y_pred_baseline = baseline_model.predict(X_text_only)

    # 텍스트 + 그래프 (Full)
    X_full = np.hstack([
        text_emb[sample_indices],
        graph_emb[sample_indices],
        graph_feat.iloc[sample_indices].values
    ])
    y_pred_full = model.predict(X_full)

    # Ground truth
    y_true = df.iloc[sample_indices]['bias_initial'].values

    # 성능 비교
    baseline_mae = mean_absolute_error(y_true, y_pred_baseline)
    full_mae     = mean_absolute_error(y_true, y_pred_full)
    improvement  = baseline_mae - full_mae
    improvement_pct = (improvement / baseline_mae) * 100

    print(f"\n  Overall Performance:")
    print(f"    - Baseline MAE: {baseline_mae:.2f}")
    print(f"    - Full MAE:     {full_mae:.2f}")
    print(f"    - Improvement:  {improvement:.2f} ({improvement_pct:.1f}%)")

    # 샘플별 상세 분석
    print(f"\n  Sample-level Analysis:")

    better_count = 0
    worse_count  = 0
    same_count   = 0

    for i in range(n_samples):
        baseline_error = abs(y_pred_baseline[i] - y_true[i])
        full_error     = abs(y_pred_full[i] - y_true[i])

        if full_error < baseline_error - 0.5:
            better_count += 1
        elif full_error > baseline_error + 0.5:
            worse_count += 1
        else:
            same_count += 1

    print(f"    - Better: {better_count} ({better_count/n_samples*100:.1f}%)")
    print(f"    - Same:   {same_count} ({same_count/n_samples*100:.1f}%)")
    print(f"    - Worse:  {worse_count} ({worse_count/n_samples*100:.1f}%)")

    # 그래프 Feature 통계
    print(f"\n  Graph Feature Statistics (Sample):")
    graph_sample = graph_feat.iloc[sample_indices]
    print(graph_sample.describe().to_string())

    # has_opposite 영향 분석
    print(f"\n  Impact of 'has_opposite' Feature:")
    has_opposite_mask = graph_sample['has_opposite'].values == 1

    if has_opposite_mask.sum() > 0:
        baseline_mae_with = mean_absolute_error(
            y_true[has_opposite_mask], y_pred_baseline[has_opposite_mask])
        full_mae_with = mean_absolute_error(
            y_true[has_opposite_mask], y_pred_full[has_opposite_mask])
        baseline_mae_without = mean_absolute_error(
            y_true[~has_opposite_mask], y_pred_baseline[~has_opposite_mask])
        full_mae_without = mean_absolute_error(
            y_true[~has_opposite_mask], y_pred_full[~has_opposite_mask])

        print(f"    With opposite stance (n={has_opposite_mask.sum()}):")
        print(f"      - Baseline MAE: {baseline_mae_with:.2f}")
        print(f"      - Full MAE:     {full_mae_with:.2f}")
        print(f"      - Improvement:  {baseline_mae_with - full_mae_with:.2f}")

        print(f"    Without opposite stance (n={(~has_opposite_mask).sum()}):")
        print(f"      - Baseline MAE: {baseline_mae_without:.2f}")
        print(f"      - Full MAE:     {full_mae_without:.2f}")
        print(f"      - Improvement:  {baseline_mae_without - full_mae_without:.2f}")

    # neighbor_avg_bias 상관관계 분석
    print(f"\n  Correlation Analysis:")
    corr_true, p_true = pearsonr(graph_sample['neighbor_avg_bias'], y_true)
    corr_pred, p_pred = pearsonr(graph_sample['neighbor_avg_bias'], y_pred_full)

    print(f"    - neighbor_avg_bias vs true bias:")
    print(f"      r={corr_true:.3f}, p={p_true:.4f}")
    print(f"    - neighbor_avg_bias vs full prediction:")
    print(f"      r={corr_pred:.3f}, p={p_pred:.4f}")

    # 개선이 가장 큰 샘플 분석
    print(f"\n  Top 5 Most Improved Samples:")
    improvements_per_sample = (y_pred_baseline - y_true)**2 - (y_pred_full - y_true)**2
    top_improved_idx = np.argsort(improvements_per_sample)[-5:][::-1]

    for rank, idx in enumerate(top_improved_idx, 1):
        sample_idx = sample_indices[idx]
        article = df.iloc[sample_idx]

        baseline_error = abs(y_pred_baseline[idx] - y_true[idx])
        full_error     = abs(y_pred_full[idx] - y_true[idx])

        print(f"\n    {rank}. {article['제목'][:50]}...")
        print(f"       Media:          {article['언론사']}")
        print(f"       True bias:      {y_true[idx]:+.1f}")
        print(f"       Baseline error: {baseline_error:.2f}")
        print(f"       Full error:     {full_error:.2f}")
        print(f"       Improvement:    {baseline_error - full_error:.2f}")
        print(f"       has_opposite:   {int(graph_sample.iloc[idx]['has_opposite'])}")
        print(f"       neighbor_avg:   {graph_sample.iloc[idx]['neighbor_avg_bias']:.1f}")


if __name__ == '__main__':
    # Full 모델 학습
    model, results = train_full_model()

    # 그래프 기여도 분석
    analyze_graph_contribution(model, n_samples=100)

    print("\nDone")