# 각 컴포넌트의 성능 기여도를 분석하는 Ablation Study

#   Model A - Text only              (768-dim)  : Baseline Model
#   Model B - Text + Graph Feature   (773-dim)  : 신규 학습
#   Model C - Text + Graph Emb       (832-dim)  : 신규 학습
#   Model D - Text + Graph Emb + Feat (837-dim) : Full Model

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_data(text_emb_path='data/embeddings/text_embeddings.npy',
              graph_emb_path='data/embeddings/graph_embeddings.npy',
              graph_feat_path='data/embeddings/graph_features.csv',
              data_path='data/processed/all_articles_labeled.csv'):
    """
    Ablation Study에 필요한 전체 데이터 로드

    Args:
        text_emb_path: 텍스트 임베딩 경로 (479188, 768)
        graph_emb_path: 그래프 임베딩 경로 (479188, 64)
        graph_feat_path: 그래프 Feature 경로 (479188, 5)
        data_path: 레이블 데이터 경로

    Returns:
        text_emb: 텍스트 임베딩
        graph_emb: 그래프 임베딩
        graph_feat: 그래프 Feature DataFrame
        y: 편향도 레이블
    """

    print("Loading data...")
    text_emb   = np.load(text_emb_path)
    graph_emb  = np.load(graph_emb_path)
    graph_feat = pd.read_csv(graph_feat_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    y  = df['bias_initial'].values

    print(f"  - Text embeddings:  {text_emb.shape}")
    print(f"  - Graph embeddings: {graph_emb.shape}")
    print(f"  - Graph features:   {graph_feat.shape}")
    print(f"  - Labels:           {len(y):,}")

    return text_emb, graph_emb, graph_feat, y


def build_splits(text_emb, y, test_size=0.3, val_size=0.5, random_state=42):
    """
    학습 시와 동일한 파라미터로 인덱스 분할
    모든 모델이 완전히 동일한 train/val/test 샘플을 사용하도록 보장

    Args:
        text_emb: 전체 데이터 (크기 확인용)
        y: 레이블
        test_size, val_size, random_state: 학습 시 사용한 파라미터와 동일해야 함

    Returns:
        train_idx, val_idx, test_idx: 분할 인덱스
    """

    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_size, random_state=random_state
    )

    print(f"\nData split (same as training):")
    print(f"  - Train: {len(train_idx):,} ({len(train_idx)/len(y)*100:.1f}%)")
    print(f"  - Val:   {len(val_idx):,} ({len(val_idx)/len(y)*100:.1f}%)")
    print(f"  - Test:  {len(test_idx):,} ({len(test_idx)/len(y)*100:.1f}%)")

    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────
# 단일 모델 학습 및 평가
# ──────────────────────────────────────────────

def train_and_evaluate(X, y, train_idx, val_idx, test_idx,
                       model_name, random_state=42):
    """
    단일 Feature 조합으로 모델 학습 및 평가

    Args:
        X: 입력 Feature 행렬
        y: 레이블
        train_idx, val_idx, test_idx: 분할 인덱스
        model_name: 출력용 모델 이름
        random_state: 재현성 시드

    Returns:
        metrics: 성능 지표 딕셔너리
    """

    print(f"\n  [{model_name}]  input dim: {X.shape[1]}")

    X_train = X[train_idx]
    X_val   = X[val_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0       # Ablation Study는 개별 트리 로그 생략
    )

    start = datetime.now()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = (datetime.now() - start).total_seconds()

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"    MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}  "
          f"(time: {elapsed:.1f}s)")

    return {
        'model_name': model_name,
        'input_dim': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_time': elapsed,
    }


# ──────────────────────────────────────────────
# Ablation Study 메인
# ──────────────────────────────────────────────

def ablation_study(text_emb_path='data/embeddings/text_embeddings.npy',
                   graph_emb_path='data/embeddings/graph_embeddings.npy',
                   graph_feat_path='data/embeddings/graph_features.csv',
                   data_path='data/processed/all_articles_labeled.csv',
                   baseline_results_path='data/models/baseline_results.pkl',
                   full_results_path='data/models/full_results.pkl',
                   save_dir='data/models',
                   test_size=0.3,
                   val_size=0.5,
                   random_state=42):
    """
    4가지 Feature 조합의 성능을 비교하여 각 컴포넌트의 기여도 측정

    Model A - Text only              (768-dim) : Baseline Model
    Model B - Text + Graph Feature   (773-dim) : 신규 학습
    Model C - Text + Graph Emb       (832-dim) : 신규 학습
    Model D - Text + Graph Emb + Feat (837-dim): Full Model

    Args:
        text_emb_path: 텍스트 임베딩 경로
        graph_emb_path: 그래프 임베딩 경로
        graph_feat_path: 그래프 Feature 경로
        data_path: 레이블 데이터 경로
        baseline_results_path: baseline 모델 경로
        full_results_path: full 모델 경로
        save_dir: 결과 저장 디렉토리
        test_size, val_size, random_state: 학습 시와 동일한 split 파라미터

    Returns:
        results_df: 모델별 성능 DataFrame
    """

    print("="*80)
    print("Ablation Study: Component-wise Contribution Analysis")
    print("="*80)
    print("\nComparing 4 feature combinations:")
    print("  Model A - Text only              (768-dim)")
    print("  Model B - Text + Graph Feature   (773-dim)")
    print("  Model C - Text + Graph Emb       (832-dim)")
    print("  Model D - Text + Graph Emb + Feat (837-dim)")

    # 데이터 로드
    print("\n" + "-"*80)
    text_emb, graph_emb, graph_feat, y = load_data(
        text_emb_path, graph_emb_path, graph_feat_path, data_path
    )

    # 동일한 Split 구성
    print("\n" + "-"*80)
    train_idx, val_idx, test_idx = build_splits(
        text_emb, y, test_size, val_size, random_state
    )

    # Feature 조합별 학습 및 평가
    print("\n" + "-"*80)
    print("Training and evaluating each model configuration...")

    results = []

    # Model A: Text only
    if os.path.exists(baseline_results_path):
        print(f"\n  [Model A - Text only]  input dim: 768")
        with open(baseline_results_path, 'rb') as f:
            saved = pickle.load(f)
        # baseline_results.pkl 구조에서 test 지표 추출
        metrics_a = {
            'model_name': 'A - Text only',
            'input_dim': 768,
            'mae':  saved.get('test_mae',  saved.get('mae')),
            'rmse': saved.get('test_rmse', saved.get('rmse')),
            'r2':   saved.get('test_r2',   saved.get('r2')),
            'train_time': saved.get('train_time', 0),
        }
        print(f"    MAE: {metrics_a['mae']:.4f}  RMSE: {metrics_a['rmse']:.4f}"
              f"  R²: {metrics_a['r2']:.4f}")
    else:
        print(f"\n  [Model A] baseline_results.pkl not found, training from scratch...")
        X_a = text_emb
        metrics_a = train_and_evaluate(
            X_a, y, train_idx, val_idx, test_idx, 'A - Text only', random_state
        )
    results.append(metrics_a)

    # Model B: Text + Graph Feature
    X_b = np.hstack([text_emb, graph_feat.values])
    metrics_b = train_and_evaluate(
        X_b, y, train_idx, val_idx, test_idx, 'B - Text + Graph Feat', random_state
    )
    results.append(metrics_b)

    # Model C: Text + Graph Emb
    X_c = np.hstack([text_emb, graph_emb])
    metrics_c = train_and_evaluate(
        X_c, y, train_idx, val_idx, test_idx, 'C - Text + Graph Emb', random_state
    )
    results.append(metrics_c)

    # Model D: Text + Graph Emb + Graph Feature
    if os.path.exists(full_results_path):
        print(f"\n  [Model D - Text + Graph Emb + Feat]  input dim: 837")
        with open(full_results_path, 'rb') as f:
            saved = pickle.load(f)
        metrics_d = {
            'model_name': 'D - Text + Graph Emb + Feat',
            'input_dim': 837,
            'mae':  saved.get('test_mae',  saved.get('mae')),
            'rmse': saved.get('test_rmse', saved.get('rmse')),
            'r2':   saved.get('test_r2',   saved.get('r2')),
            'train_time': saved.get('train_time', 0),
        }
        print(f"    MAE: {metrics_d['mae']:.4f}  RMSE: {metrics_d['rmse']:.4f}"
              f"  R²: {metrics_d['r2']:.4f}")
    else:
        print(f"\n  [Model D] full_results.pkl not found, training from scratch...")
        X_d = np.hstack([text_emb, graph_emb, graph_feat.values])
        metrics_d = train_and_evaluate(
            X_d, y, train_idx, val_idx, test_idx,
            'D - Text + Graph Emb + Feat', random_state
        )
    results.append(metrics_d)

    # 결과 정리
    results_df = pd.DataFrame(results)

    # 기여도 계산 (Model A 대비 개선량)
    baseline_mae = results_df.loc[results_df['model_name'] == 'A - Text only', 'mae'].values[0]
    results_df['mae_reduction']     = baseline_mae - results_df['mae']
    results_df['mae_reduction_pct'] = results_df['mae_reduction'] / baseline_mae * 100

    # 결과 출력
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    print(f"\n{'Model':<30} {'Dim':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
          f"{'MAE Drop':>10} {'Drop %':>8}")
    print("-"*82)
    for _, row in results_df.iterrows():
        print(f"  {row['model_name']:<28} {int(row['input_dim']):>6} "
              f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f} "
              f"{row['mae_reduction']:>+10.4f} {row['mae_reduction_pct']:>+7.1f}%")
    print("-"*82)

    # 컴포넌트별 기여도 분석
    print("\nComponent Contribution Analysis:")

    mae_a = results_df[results_df['model_name'] == 'A - Text only']['mae'].values[0]
    mae_b = results_df[results_df['model_name'] == 'B - Text + Graph Feat']['mae'].values[0]
    mae_c = results_df[results_df['model_name'] == 'C - Text + Graph Emb']['mae'].values[0]
    mae_d = results_df[results_df['model_name'] == 'D - Text + Graph Emb + Feat']['mae'].values[0]

    contrib_feat = mae_a - mae_b   # Graph Feature 단독 기여
    contrib_emb  = mae_a - mae_c   # Graph Emb 단독 기여
    contrib_full = mae_a - mae_d   # 전체 그래프 기여

    print(f"  Graph Feature alone (A -> B): {contrib_feat:>+.4f} MAE reduction")
    print(f"  Graph Emb alone     (A -> C): {contrib_emb:>+.4f} MAE reduction")
    print(f"  Both combined       (A -> D): {contrib_full:>+.4f} MAE reduction")

    if contrib_emb > contrib_feat:
        print(f"\n  Graph Emb contributes more than Graph Feature")
    else:
        print(f"\n  Graph Feature contributes more than Graph Emb")

    synergy = contrib_full - contrib_feat - contrib_emb
    print(f"  Synergy effect (interaction): {synergy:>+.4f}")

    # 저장
    print("\n" + "-"*80)
    print("Saving results...")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, 'ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")

    return results_df


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def visualize_ablation(results_df, save_dir='data/models'):
    """
    Ablation Study 결과 시각화 (3개 서브플롯)

    Args:
        results_df: ablation_study() 반환값
        save_dir: 저장 디렉토리
    """

    print("\nGenerating visualization...")

    labels = [r.split(' - ')[1] for r in results_df['model_name'].tolist()]
    maes   = results_df['mae'].tolist()
    r2s    = results_df['r2'].tolist()
    drops  = results_df['mae_reduction'].tolist()
    colors = ['steelblue', 'mediumseagreen', 'mediumpurple', 'coral']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ablation Study: Component Contribution Analysis', fontsize=14)

    # MAE 비교
    bars = axes[0].bar(labels, maes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[0].set_ylabel('Test MAE')
    axes[0].set_title('MAE by Model Configuration')
    axes[0].set_ylim(0, max(maes) * 1.2)
    axes[0].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # R² 비교
    bars = axes[1].bar(labels, r2s, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[1].set_ylabel('R-squared')
    axes[1].set_title('R² by Model Configuration')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, r2s):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # MAE Reduction (Baseline 대비)
    bar_colors = ['gray'] + ['coral' if d > 0 else 'lightcoral' for d in drops[1:]]
    bars = axes[2].bar(labels, drops, color=bar_colors, alpha=0.85,
                       edgecolor='black', linewidth=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('MAE Reduction vs Text-only')
    axes[2].set_title('MAE Reduction over Baseline')
    axes[2].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, drops):
        va = 'bottom' if val >= 0 else 'top'
        axes[2].text(bar.get_x() + bar.get_width()/2, val,
                     f'{val:+.3f}', ha='center', va=va, fontsize=9)

    plt.tight_layout()

    img_path = os.path.join(save_dir, 'ablation_study.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {img_path}")


if __name__ == '__main__':
    # Ablation Study 실행
    results_df = ablation_study()

    # 시각화
    visualize_ablation(results_df)

    print("\nDone")