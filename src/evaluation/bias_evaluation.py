# 편향도 예측 모델의 성능을 언론사별, 편향도 구간별로 상세 분석

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_data(data_path='data/processed/all_articles_with_bias.csv'):
    """
    전체 기사 + 예측 편향도 데이터 로드

    Args:
        data_path: all_articles_with_bias.csv 경로

    Returns:
        df: bias_initial, bias_predicted 컬럼이 포함된 DataFrame
    """

    print("Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}\n")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"  Loaded: {len(df):,} articles")
    print(f"  Columns: {df.columns.tolist()}")

    return df


# ──────────────────────────────────────────────
# 전체 성능 평가
# ──────────────────────────────────────────────

def evaluate_overall(df):
    """
    전체 데이터에 대한 종합 성능 지표 계산

    Args:
        df: bias_initial, bias_predicted 컬럼 포함 DataFrame

    Returns:
        metrics: 성능 지표 딕셔너리
    """

    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)

    y_true = df['bias_initial'].values
    y_pred = df['bias_predicted'].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    errors = y_pred - y_true

    print(f"\nMetrics (all {len(df):,} articles):")
    print(f"  - MAE:         {mae:.4f}")
    print(f"  - RMSE:        {rmse:.4f}")
    print(f"  - R-squared:   {r2:.4f}")
    print(f"  - Correlation: {corr:.4f}")

    print(f"\nResidual Statistics:")
    print(f"  - Mean error:   {errors.mean():>+.4f}  (positive = overestimate)")
    print(f"  - Std error:    {errors.std():>.4f}")
    print(f"  - Median error: {np.median(errors):>+.4f}")
    print(f"  - Max error:    {errors.max():>+.4f}")
    print(f"  - Min error:    {errors.min():>+.4f}")

    # 오차 구간별 비율
    abs_errors = np.abs(errors)
    print(f"\nError Distribution:")
    for t in [5, 10, 20, 30]:
        pct = (abs_errors < t).mean() * 100
        print(f"  - Error < {t:2d}: {pct:.1f}%")

    metrics = {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'corr': corr,
        'mean_error': errors.mean(), 'std_error': errors.std(),
    }
    return metrics


# ──────────────────────────────────────────────
# 언론사별 평가
# ──────────────────────────────────────────────

def evaluate_by_media(df, top_n=10):
    """
    언론사별 MAE, RMSE, 기사 수 분석

    Args:
        df: 전체 DataFrame
        top_n: 상위/하위 출력 언론사 수

    Returns:
        media_stats: 언론사별 성능 DataFrame (MAE 오름차순 정렬)
    """

    print("\n" + "="*80)
    print("PERFORMANCE BY MEDIA")
    print("="*80)

    records = []
    for media, group in df.groupby('언론사'):
        y_true = group['bias_initial'].values
        y_pred = group['bias_predicted'].values
        records.append({
            'media':    media,
            'n':        len(group),
            'mae':      mean_absolute_error(y_true, y_pred),
            'rmse':     np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2':       r2_score(y_true, y_pred) if len(group) > 1 else np.nan,
            'mean_pred': y_pred.mean(),
            'mean_true': y_true.mean(),
        })

    media_stats = pd.DataFrame(records).sort_values('mae').reset_index(drop=True)

    # 상위/하위 출력
    print(f"\n{'Rank':<5} {'Media':<22} {'N':>7} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-"*62)

    print(f"  [Best {top_n} - Lowest MAE]")
    for i, row in media_stats.head(top_n).iterrows():
        print(f"  {i+1:<4} {row['media']:<22} {int(row['n']):>7,} "
              f"{row['mae']:>8.3f} {row['rmse']:>8.3f} {row['r2']:>8.3f}")

    print(f"\n  [Worst {top_n} - Highest MAE]")
    for i, row in media_stats.tail(top_n).iterrows():
        print(f"  {i+1:<4} {row['media']:<22} {int(row['n']):>7,} "
              f"{row['mae']:>8.3f} {row['rmse']:>8.3f} {row['r2']:>8.3f}")

    # 전체 통계
    print(f"\nSummary:")
    print(f"  - Total media outlets:   {len(media_stats)}")
    print(f"  - Mean MAE (per media):  {media_stats['mae'].mean():.4f}")
    print(f"  - Std  MAE (per media):  {media_stats['mae'].std():.4f}")
    print(f"  - Median MAE:            {media_stats['mae'].median():.4f}")

    return media_stats


# ──────────────────────────────────────────────
# 편향도 구간별 평가
# ──────────────────────────────────────────────

def evaluate_by_bias_range(df):
    """
    편향도 구간별 (극진보 / 진보 / 중립 / 보수 / 극보수) 예측 정확도 분석

    Args:
        df: 전체 DataFrame

    Returns:
        range_stats: 구간별 성능 DataFrame
    """

    print("\n" + "="*80)
    print("PERFORMANCE BY BIAS RANGE")
    print("="*80)

    bins   = [-101, -50, -20, 20, 50, 101]
    labels = ['Far-Progressive', 'Progressive', 'Neutral', 'Conservative', 'Far-Conservative']

    df = df.copy()
    df['bias_range'] = pd.cut(df['bias_initial'], bins=bins, labels=labels)

    records = []
    for label in labels:
        group = df[df['bias_range'] == label]
        if len(group) == 0:
            continue
        y_true = group['bias_initial'].values
        y_pred = group['bias_predicted'].values
        records.append({
            'bias_range': label,
            'n':          len(group),
            'pct':        len(group) / len(df) * 100,
            'mae':        mean_absolute_error(y_true, y_pred),
            'rmse':       np.sqrt(mean_squared_error(y_true, y_pred)),
            'mean_error': (y_pred - y_true).mean(),
        })

    range_stats = pd.DataFrame(records)

    print(f"\n{'Bias Range':<20} {'N':>8} {'%':>6} {'MAE':>8} {'RMSE':>8} {'Mean Err':>10}")
    print("-"*64)
    for _, row in range_stats.iterrows():
        print(f"  {row['bias_range']:<18} {int(row['n']):>8,} {row['pct']:>5.1f}% "
              f"{row['mae']:>8.3f} {row['rmse']:>8.3f} {row['mean_error']:>+10.3f}")

    # 가장 어려운 구간
    hardest = range_stats.loc[range_stats['mae'].idxmax(), 'bias_range']
    easiest = range_stats.loc[range_stats['mae'].idxmin(), 'bias_range']
    print(f"\n  Hardest range (highest MAE): {hardest}")
    print(f"  Easiest range (lowest MAE):  {easiest}")

    return range_stats


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def visualize(df, media_stats, range_stats, save_dir='data/models'):
    """
    4개 서브플롯으로 평가 결과 시각화

    (1) Scatter: 실제 편향도 vs 예측 편향도
    (2) 오차 분포 히스토그램
    (3) 언론사별 MAE (상위 15개)
    (4) 편향도 구간별 MAE

    Args:
        df: 전체 DataFrame
        media_stats: evaluate_by_media() 반환값
        range_stats: evaluate_by_bias_range() 반환값
        save_dir: 저장 디렉토리
    """

    print("\nGenerating visualization...")

    y_true  = df['bias_initial'].values
    y_pred  = df['bias_predicted'].values
    errors  = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bias Prediction Model Evaluation', fontsize=15)

    # (1) Scatter: True vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.05, s=1, color='steelblue')
    ax.plot([-100, 100], [-100, 100], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('True Bias (bias_initial)')
    ax.set_ylabel('Predicted Bias (bias_predicted)')
    ax.set_title('True vs Predicted Bias')
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)
    ax.legend(fontsize=9)

    mae_val  = mean_absolute_error(y_true, y_pred)
    r2_val   = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f'MAE={mae_val:.2f}  R²={r2_val:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # (2) 오차 분포 히스토그램
    ax = axes[0, 1]
    ax.hist(errors, bins=100, alpha=0.75, color='steelblue', edgecolor='none')
    ax.axvline(0,             color='red',    linestyle='--', linewidth=1.5, label='Zero error')
    ax.axvline(errors.mean(), color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean={errors.mean():+.2f}')
    ax.set_xlabel('Prediction Error (predicted - true)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend(fontsize=9)

    # (3) 언론사별 MAE (상위 15개 - 가장 낮은)
    ax = axes[1, 0]
    top15 = media_stats.head(15)
    ax.barh(top15['media'], top15['mae'], color='mediumseagreen',
            alpha=0.85, edgecolor='none')
    ax.set_xlabel('MAE')
    ax.set_title('Top 15 Media - Lowest MAE')
    ax.invert_yaxis()
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.text(row['mae'], i, f" {row['mae']:.2f}", va='center', fontsize=8)

    # (4) 편향도 구간별 MAE
    ax = axes[1, 1]
    colors_range = ['#4575b4', '#74add1', '#abd9e9', '#f46d43', '#d73027']
    bars = ax.bar(range_stats['bias_range'], range_stats['mae'],
                  color=colors_range[:len(range_stats)], alpha=0.85, edgecolor='none')
    ax.set_xlabel('Bias Range')
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Bias Range')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, range_stats['mae']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    img_path = os.path.join(save_dir, 'bias_evaluation.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {img_path}")


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────

def save_results(media_stats, range_stats, save_dir='data/models'):
    """
    언론사별 MAE 테이블 저장

    Args:
        media_stats: evaluate_by_media() 반환값
        range_stats: evaluate_by_bias_range() 반환값
        save_dir: 저장 디렉토리
    """

    print("\nSaving results...")
    os.makedirs(save_dir, exist_ok=True)

    # 언론사별 MAE
    media_path = os.path.join(save_dir, 'media_mae.csv')
    media_stats.to_csv(media_path, index=False, encoding='utf-8-sig')
    print(f"  Media MAE saved: {media_path}")

    # 편향도 구간별 성능
    range_path = os.path.join(save_dir, 'range_mae.csv')
    range_stats.to_csv(range_path, index=False, encoding='utf-8-sig')
    print(f"  Range MAE saved: {range_path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def evaluate_bias_model(data_path='data/processed/all_articles_with_bias.csv',
                        save_dir='data/models'):
    """
    편향도 예측 모델 상세 평가 실행

    Args:
        data_path: 데이터 경로 
        save_dir: 결과 저장 디렉토리

    Returns:
        media_stats: 언론사별 성능 DataFrame
        range_stats: 구간별 성능 DataFrame
    """

    print("="*80)
    print("Bias Prediction Model Evaluation")
    print("="*80)

    # 데이터 로드
    df = load_data(data_path)

    # 전체 성능
    metrics = evaluate_overall(df)

    # 언론사별 성능
    media_stats = evaluate_by_media(df, top_n=10)

    # 편향도 구간별 성능
    range_stats = evaluate_by_bias_range(df)

    # 시각화
    visualize(df, media_stats, range_stats, save_dir)

    # 저장
    save_results(media_stats, range_stats, save_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"\n  Overall MAE:  {metrics['mae']:.4f}")
    print(f"  Overall R²:   {metrics['r2']:.4f}")
    print(f"\n  Saved files:")
    print(f"    {save_dir}/media_mae.csv")
    print(f"    {save_dir}/range_mae.csv")
    print(f"    {save_dir}/bias_evaluation.png")
    print("="*80)

    return media_stats, range_stats


if __name__ == '__main__':
    media_stats, range_stats = evaluate_bias_model()

    print("\nDone")