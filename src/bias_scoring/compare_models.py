# 학습된 Baseline 모델과 Full 모델의 성능을 로드하여 비교

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle
import os
from datetime import datetime


def load_models(baseline_model_path='data/models/bias_model_baseline.pkl',
                full_model_path='data/models/bias_model_full.pkl'):
    """
    학습된 Baseline 모델과 Full 모델 로드

    Args:
        baseline_model_path: Baseline 모델 경로
        full_model_path: Full 모델 경로

    Returns:
        baseline_model: 로드된 Baseline 모델
        full_model: 로드된 Full 모델
    """

    print("Loading models...")

    # Baseline 모델 로드
    if not os.path.exists(baseline_model_path):
        raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}\n")
    with open(baseline_model_path, 'rb') as f:
        baseline_model = pickle.load(f)
    print(f"  Baseline model loaded: {baseline_model_path}")

    # Full 모델 로드
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Full model not found: {full_model_path}\n")
    with open(full_model_path, 'rb') as f:
        full_model = pickle.load(f)
    print(f"  Full model loaded:     {full_model_path}")

    return baseline_model, full_model


def load_data(text_emb_path='data/embeddings/text_embeddings.npy',
              graph_emb_path='data/embeddings/graph_embeddings.npy',
              graph_feat_path='data/embeddings/graph_features.csv',
              data_path='data/processed/all_articles_labeled.csv'):
    """
    비교 평가에 필요한 데이터 로드

    Args:
        text_emb_path: 텍스트 임베딩 경로 (479188, 768)
        graph_emb_path: 그래프 임베딩 경로 (479188, 64)
        graph_feat_path: 그래프 Feature 경로 (479188, 5)
        data_path: 레이블 데이터 경로

    Returns:
        text_emb: 텍스트 임베딩
        graph_emb: 그래프 임베딩
        graph_feat: 그래프 Feature DataFrame
        df: 전체 기사 DataFrame
        y: 편향도 레이블
    """

    print("\nLoading data...")
    text_emb   = np.load(text_emb_path)
    graph_emb  = np.load(graph_emb_path)
    graph_feat = pd.read_csv(graph_feat_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    y  = df['bias_initial'].values

    print(f"  - Text embeddings:  {text_emb.shape}")
    print(f"  - Graph embeddings: {graph_emb.shape}")
    print(f"  - Graph features:   {graph_feat.shape}")
    print(f"  - Labels:           {len(y)}")

    return text_emb, graph_emb, graph_feat, df, y


def build_test_set(text_emb, graph_emb, graph_feat, y,
                   test_size=0.3, val_size=0.5, random_state=42):
    """
    학습 시와 동일한 random_state로 Test set 재구성
    (train_baseline_model.py, train_full_model.py와 동일한 split 파라미터 사용)

    Args:
        text_emb: 텍스트 임베딩
        graph_emb: 그래프 임베딩
        graph_feat: 그래프 Feature
        y: 레이블
        test_size: 학습 시 사용한 test_size
        val_size: 학습 시 사용한 val_size
        random_state: 학습 시 사용한 random_state

    Returns:
        X_test_text: Baseline용 Test 입력 (N_test, 768)
        X_test_full: Full모델용 Test 입력 (N_test, 837)
        y_test: Test 레이블
        test_idx: Test 인덱스 (언론사별 분석용)
    """
    from sklearn.model_selection import train_test_split

    print("\nReconstructing test set (same split as training)...")

    indices = np.arange(len(y))
    _, temp_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    _, test_idx = train_test_split(temp_idx, test_size=val_size,  random_state=random_state)

    X_test_text = text_emb[test_idx]
    X_test_full = np.hstack([text_emb[test_idx],
                              graph_emb[test_idx],
                              graph_feat.iloc[test_idx].values])
    y_test = y[test_idx]

    print(f"  - Test samples: {len(y_test):,} ({len(y_test)/len(y)*100:.1f}%)")
    print(f"  - Baseline input shape: {X_test_text.shape}")
    print(f"  - Full model input shape: {X_test_full.shape}")

    return X_test_text, X_test_full, y_test, test_idx


def compare_performance(baseline_model, full_model,
                        X_test_text, X_test_full, y_test):
    """
    동일한 Test set에서 두 모델의 성능 지표 비교

    Args:
        baseline_model: Baseline 모델
        full_model: Full 모델
        X_test_text: Baseline용 Test 입력
        X_test_full: Full 모델용 Test 입력
        y_test: Test 레이블

    Returns:
        baseline_metrics: Baseline 성능 지표
        full_metrics: Full 모델 성능 지표
        y_pred_baseline: Baseline 예측값
        y_pred_full: Full 모델 예측값
    """

    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)

    # 예측
    y_pred_baseline = baseline_model.predict(X_test_text)
    y_pred_full     = full_model.predict(X_test_full)

    # Baseline 지표
    baseline_metrics = {
        'mae':  mean_absolute_error(y_test, y_pred_baseline),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
        'r2':   r2_score(y_test, y_pred_baseline),
        'corr': pearsonr(y_test, y_pred_baseline)[0],
    }

    # Full 모델 지표
    full_metrics = {
        'mae':  mean_absolute_error(y_test, y_pred_full),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_full)),
        'r2':   r2_score(y_test, y_pred_full),
        'corr': pearsonr(y_test, y_pred_full)[0],
    }

    # 개선량
    improvement_mae  = baseline_metrics['mae']  - full_metrics['mae']
    improvement_pct  = improvement_mae / baseline_metrics['mae'] * 100
    improvement_r2   = full_metrics['r2'] - baseline_metrics['r2']
    improvement_corr = full_metrics['corr'] - baseline_metrics['corr']

    # 출력
    print(f"\n{'Metric':<12} {'Baseline':>12} {'Full Model':>12} {'Improvement':>14}")
    print("-"*52)
    print(f"{'MAE':<12} {baseline_metrics['mae']:>12.4f} {full_metrics['mae']:>12.4f} "
          f"{improvement_mae:>+13.4f}")
    print(f"{'RMSE':<12} {baseline_metrics['rmse']:>12.4f} {full_metrics['rmse']:>12.4f} "
          f"{baseline_metrics['rmse']-full_metrics['rmse']:>+13.4f}")
    print(f"{'R-squared':<12} {baseline_metrics['r2']:>12.4f} {full_metrics['r2']:>12.4f} "
          f"{improvement_r2:>+13.4f}")
    print(f"{'Correlation':<12} {baseline_metrics['corr']:>12.4f} {full_metrics['corr']:>12.4f} "
          f"{improvement_corr:>+13.4f}")
    print("-"*52)
    print(f"\nMAE Reduction:  {improvement_mae:.4f} ({improvement_pct:+.1f}%)")

    # 목표 달성 여부
    print(f"\nTarget Achievement:")
    print(f"  Baseline  (MAE < 25): {'ACHIEVED' if baseline_metrics['mae'] < 25 else 'NOT MET'}"
          f"  (MAE = {baseline_metrics['mae']:.2f})")
    print(f"  Full Model (MAE < 20): {'ACHIEVED' if full_metrics['mae'] < 20 else 'NOT MET'}"
          f"  (MAE = {full_metrics['mae']:.2f})")

    return baseline_metrics, full_metrics, y_pred_baseline, y_pred_full


def compare_by_media(df, test_idx, y_test, y_pred_baseline, y_pred_full, top_n=10):
    """
    언론사별 두 모델의 성능 비교

    Args:
        df: 전체 기사 DataFrame
        test_idx: Test 인덱스
        y_test: Test 레이블
        y_pred_baseline: Baseline 예측값
        y_pred_full: Full 모델 예측값
        top_n: 출력할 언론사 수
    """

    print("\n" + "="*80)
    print("PERFORMANCE BY MEDIA")
    print("="*80)

    test_df = df.iloc[test_idx].copy()
    test_df['true']              = y_test
    test_df['pred_baseline']     = y_pred_baseline
    test_df['pred_full']         = y_pred_full
    test_df['error_baseline']    = np.abs(y_pred_baseline - y_test)
    test_df['error_full']        = np.abs(y_pred_full - y_test)

    # 언론사별 집계
    media_stats = test_df.groupby('언론사').agg(
        n=('true', 'count'),
        baseline_mae=('error_baseline', 'mean'),
        full_mae=('error_full', 'mean'),
    ).reset_index()
    media_stats['improvement'] = media_stats['baseline_mae'] - media_stats['full_mae']
    media_stats['improvement_pct'] = (media_stats['improvement']
                                      / media_stats['baseline_mae'] * 100)

    # Full 모델 기준 MAE 정렬
    media_stats = media_stats.sort_values('full_mae')

    # 상위/하위 출력
    print(f"\n{'Media':<20} {'N':>6} {'Baseline MAE':>13} {'Full MAE':>10} {'Improvement':>13}")
    print("-"*66)

    print(f"  [Best {top_n} - Lowest Full MAE]")
    for _, row in media_stats.head(top_n).iterrows():
        print(f"  {row['언론사']:<20} {int(row['n']):>6} "
              f"{row['baseline_mae']:>13.2f} {row['full_mae']:>10.2f} "
              f"{row['improvement']:>+10.2f} ({row['improvement_pct']:>+5.1f}%)")

    print(f"\n  [Worst {top_n} - Highest Full MAE]")
    for _, row in media_stats.tail(top_n).iterrows():
        print(f"  {row['언론사']:<20} {int(row['n']):>6} "
              f"{row['baseline_mae']:>13.2f} {row['full_mae']:>10.2f} "
              f"{row['improvement']:>+10.2f} ({row['improvement_pct']:>+5.1f}%)")

    # 전체 통계
    print(f"\nOverall Media Statistics:")
    print(f"  - Total media outlets: {len(media_stats)}")
    print(f"  - Media where Full > Baseline: "
          f"{(media_stats['improvement'] > 0).sum()} / {len(media_stats)}")
    print(f"  - Mean improvement per media: {media_stats['improvement'].mean():.2f}")

    return media_stats


def compare_error_distribution(y_test, y_pred_baseline, y_pred_full):
    """
    잔차(residual) 분포 비교

    Args:
        y_test: 실제값
        y_pred_baseline: Baseline 예측값
        y_pred_full: Full 모델 예측값
    """

    print("\n" + "="*80)
    print("ERROR DISTRIBUTION COMPARISON")
    print("="*80)

    errors_baseline = np.abs(y_pred_baseline - y_test)
    errors_full     = np.abs(y_pred_full - y_test)

    thresholds = [5, 10, 20, 30]

    print(f"\n{'Error Threshold':<18} {'Baseline (%)':>14} {'Full Model (%)':>15} {'Diff':>8}")
    print("-"*58)
    for t in thresholds:
        b_pct = (errors_baseline < t).mean() * 100
        f_pct = (errors_full     < t).mean() * 100
        print(f"  Error < {t:<8}   {b_pct:>13.1f}% {f_pct:>14.1f}% {f_pct-b_pct:>+7.1f}%")

    print(f"\nResidual Statistics:")
    print(f"  {'':18} {'Baseline':>12} {'Full Model':>12}")
    print(f"  {'Mean error':18} {errors_baseline.mean():>12.2f} {errors_full.mean():>12.2f}")
    print(f"  {'Std error':18} {errors_baseline.std():>12.2f} {errors_full.std():>12.2f}")
    print(f"  {'Median error':18} {np.median(errors_baseline):>12.2f} {np.median(errors_full):>12.2f}")
    print(f"  {'Max error':18} {errors_baseline.max():>12.2f} {errors_full.max():>12.2f}")

    # 샘플별 개선 여부
    better = (errors_full < errors_baseline - 0.5).sum()
    worse  = (errors_full > errors_baseline + 0.5).sum()
    same   = len(y_test) - better - worse

    print(f"\nSample-level Comparison (total {len(y_test):,} samples):")
    print(f"  - Full better: {better:>7,} ({better/len(y_test)*100:.1f}%)")
    print(f"  - Same:        {same:>7,} ({same/len(y_test)*100:.1f}%)")
    print(f"  - Full worse:  {worse:>7,} ({worse/len(y_test)*100:.1f}%)")


def save_comparison_results(baseline_metrics, full_metrics,
                            media_stats, save_dir='data/models'):
    """
    비교 결과 저장

    Args:
        baseline_metrics: Baseline 성능 지표
        full_metrics: Full 모델 성능 지표
        media_stats: 언론사별 성능 DataFrame
        save_dir: 저장 디렉토리
    """

    print("\nSaving comparison results...")
    os.makedirs(save_dir, exist_ok=True)

    improvement_mae = baseline_metrics['mae'] - full_metrics['mae']
    improvement_pct = improvement_mae / baseline_metrics['mae'] * 100

    # pkl 저장
    comparison = {
        'baseline': baseline_metrics,
        'full': full_metrics,
        'improvement_mae': improvement_mae,
        'improvement_pct': improvement_pct,
        'media_stats': media_stats,
        'timestamp': datetime.now().isoformat(),
    }
    results_path = os.path.join(save_dir, 'model_comparison.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(comparison, f)
    print(f"  Results saved: {results_path}")

    # 텍스트 요약 저장
    summary_path = os.path.join(save_dir, 'model_comparison.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{'Metric':<12} {'Baseline':>12} {'Full Model':>12} {'Improvement':>14}\n")
        f.write("-"*52 + "\n")
        f.write(f"{'MAE':<12} {baseline_metrics['mae']:>12.4f} "
                f"{full_metrics['mae']:>12.4f} {improvement_mae:>+13.4f}\n")
        f.write(f"{'RMSE':<12} {baseline_metrics['rmse']:>12.4f} "
                f"{full_metrics['rmse']:>12.4f} "
                f"{baseline_metrics['rmse']-full_metrics['rmse']:>+13.4f}\n")
        f.write(f"{'R-squared':<12} {baseline_metrics['r2']:>12.4f} "
                f"{full_metrics['r2']:>12.4f} "
                f"{full_metrics['r2']-baseline_metrics['r2']:>+13.4f}\n")
        f.write(f"{'Correlation':<12} {baseline_metrics['corr']:>12.4f} "
                f"{full_metrics['corr']:>12.4f} "
                f"{full_metrics['corr']-baseline_metrics['corr']:>+13.4f}\n")
        f.write("-"*52 + "\n\n")
        f.write(f"MAE Reduction: {improvement_mae:.4f} ({improvement_pct:+.1f}%)\n\n")

        f.write(f"Target Achievement:\n")
        f.write(f"  Baseline   (MAE < 25): "
                f"{'ACHIEVED' if baseline_metrics['mae'] < 25 else 'NOT MET'}\n")
        f.write(f"  Full Model (MAE < 20): "
                f"{'ACHIEVED' if full_metrics['mae'] < 20 else 'NOT MET'}\n")

    print(f"  Summary saved: {summary_path}")


if __name__ == '__main__':
    print("="*80)
    print("BASELINE vs FULL MODEL COMPARISON")
    print("="*80)

    # 모델 로드
    baseline_model, full_model = load_models()

    # 데이터 로드
    text_emb, graph_emb, graph_feat, df, y = load_data()

    # 학습 시와 동일한 Test set 재구성
    X_test_text, X_test_full, y_test, test_idx = build_test_set(
        text_emb, graph_emb, graph_feat, y
    )

    # 성능 비교
    baseline_metrics, full_metrics, y_pred_baseline, y_pred_full = compare_performance(
        baseline_model, full_model,
        X_test_text, X_test_full, y_test
    )

    # 언론사별 비교
    media_stats = compare_by_media(
        df, test_idx, y_test, y_pred_baseline, y_pred_full
    )

    # 오차 분포 비교
    compare_error_distribution(y_test, y_pred_baseline, y_pred_full)

    # 결과 저장
    save_comparison_results(baseline_metrics, full_metrics, media_stats)

    print("\n" + "="*80)
    print("COMPARISON COMPLETED")
    print("="*80)
    print(f"\n  Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  Full MAE:      {full_metrics['mae']:.4f}")
    improvement = baseline_metrics['mae'] - full_metrics['mae']
    print(f"  Improvement:   {improvement:.4f} ({improvement/baseline_metrics['mae']*100:+.1f}%)")
    print(f"\n  Saved: data/models/model_comparison.pkl")
    print(f"  Saved: data/models/model_comparison.txt")
    print("\nDone")