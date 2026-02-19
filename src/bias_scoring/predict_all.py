# 학습된 Full 모델로 전체 기사의 편향도를 일괄 예측하여 저장

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime


def predict_all_bias(text_emb_path='data/embeddings/text_embeddings.npy',
                     graph_emb_path='data/embeddings/graph_embeddings.npy',
                     graph_feat_path='data/embeddings/graph_features.csv',
                     data_path='data/processed/all_articles_labeled.csv',
                     model_path='data/models/bias_model_full.pkl',
                     save_dir_emb='data/embeddings',
                     save_dir_data='data/processed'):
    """
    학습된 Full 모델로 전체 기사의 편향도를 일괄 예측

    Task 13 이후 추천 시스템에서 각 기사의 편향도를 실시간 추론이 아닌
    사전 계산된 값으로 참조하기 위해 전체 데이터에 대한 예측값을 저장

    Args:
        text_emb_path: 텍스트 임베딩 경로 (479188, 768)
        graph_emb_path: 그래프 임베딩 경로 (479188, 64)
        graph_feat_path: 그래프 Feature 경로 (479188, 5)
        data_path: 레이블 데이터 경로
        model_path: 학습된 Full 모델 경로
        save_dir_emb: 임베딩 저장 디렉토리
        save_dir_data: 데이터 저장 디렉토리

    Returns:
        bias_scores: 전체 기사 편향도 예측값 (N,)
        df: bias_predicted 컬럼이 추가된 DataFrame
    """

    print("="*80)
    print("Predicting Bias Scores for All Articles")
    print("="*80)

    # 모델 로드
    print("\nLoading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\n"
                                f"  Please run Task 9 (train_full_model.py) first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model loaded: {model_path}")

    # 데이터 로드
    print("\nLoading data...")
    text_emb   = np.load(text_emb_path)
    graph_emb  = np.load(graph_emb_path)
    graph_feat = pd.read_csv(graph_feat_path)
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    print(f"  - Text embeddings:  {text_emb.shape}")
    print(f"  - Graph embeddings: {graph_emb.shape}")
    print(f"  - Graph features:   {graph_feat.shape}")
    print(f"  - Articles:         {len(df):,}")

    # 크기 검증
    assert len(text_emb) == len(graph_emb) == len(graph_feat) == len(df), (
        f"Size mismatch: text={len(text_emb)}, graph_emb={len(graph_emb)}, "
        f"graph_feat={len(graph_feat)}, articles={len(df)}"
    )
    print(f"  Data consistency check passed")

    # Feature 결합
    print("\nCombining features...")
    X = np.hstack([text_emb, graph_emb, graph_feat.values])
    print(f"  - Combined shape: {X.shape} "
          f"({text_emb.shape[1]} + {graph_emb.shape[1]} + {graph_feat.shape[1]} = {X.shape[1]})")

    # 편향도 예측
    print("\nPredicting bias scores...")
    start_time = datetime.now()
    bias_scores = model.predict(X)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"  Prediction completed in {elapsed:.1f} seconds")
    print(f"  - Predicted articles: {len(bias_scores):,}")

    # 예측 결과 통계
    print("\nPrediction Statistics:")
    print(f"  - Range:  [{bias_scores.min():.2f}, {bias_scores.max():.2f}]")
    print(f"  - Mean:   {bias_scores.mean():.2f}")
    print(f"  - Std:    {bias_scores.std():.2f}")
    print(f"  - Median: {np.median(bias_scores):.2f}")

    # 편향 방향 분포
    n_progressive = (bias_scores < -30).sum()
    n_neutral     = ((bias_scores >= -30) & (bias_scores <= 30)).sum()
    n_conservative = (bias_scores > 30).sum()

    print(f"\nBias Distribution:")
    print(f"  - Progressive (< -30): {n_progressive:>7,} ({n_progressive/len(bias_scores)*100:.1f}%)")
    print(f"  - Neutral (-30 ~ 30):  {n_neutral:>7,} ({n_neutral/len(bias_scores)*100:.1f}%)")
    print(f"  - Conservative (> 30): {n_conservative:>7,} ({n_conservative/len(bias_scores)*100:.1f}%)")

    # 예측값 vs 초기 레이블 비교
    print("\nPredicted vs Initial Label Comparison:")
    y_initial = df['bias_initial'].values
    diff = np.abs(bias_scores - y_initial)
    print(f"  - Mean absolute diff: {diff.mean():.2f}")
    print(f"  - Std diff:           {diff.std():.2f}")
    print(f"  - Max diff:           {diff.max():.2f}")

    # 언론사별 평균 예측 편향도
    print("\nPredicted Bias by Media (Top 10 Conservative / Top 10 Progressive):")
    df_temp = df.copy()
    df_temp['bias_predicted'] = bias_scores
    media_mean = df_temp.groupby('언론사')['bias_predicted'].mean().sort_values(ascending=False)

    print("\n  Most Conservative (Highest predicted bias):")
    for media, score in media_mean.head(10).items():
        n = (df_temp['언론사'] == media).sum()
        print(f"    {media:<20}: {score:>+7.2f}  (n={n:,})")

    print("\n  Most Progressive (Lowest predicted bias):")
    for media, score in media_mean.tail(10).items():
        n = (df_temp['언론사'] == media).sum()
        print(f"    {media:<20}: {score:>+7.2f}  (n={n:,})")

    # 저장
    print("\nSaving results...")
    os.makedirs(save_dir_emb, exist_ok=True)
    os.makedirs(save_dir_data, exist_ok=True)

    # 편향도 배열 저장
    scores_path = os.path.join(save_dir_emb, 'bias_scores.npy')
    np.save(scores_path, bias_scores)
    print(f"  Bias scores saved: {scores_path}")

    # 전체 기사 + 예측 편향도 저장
    df['bias_predicted'] = bias_scores
    csv_path = os.path.join(save_dir_data, 'all_articles_with_bias.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Articles with bias saved: {csv_path}")
    print(f"  File size: {os.path.getsize(csv_path) / 1024 / 1024:.1f} MB")

    # 결과 요약
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total articles predicted: {len(bias_scores):,}")
    print(f"Prediction time:          {elapsed:.1f} seconds")
    print(f"\nBias score range: [{bias_scores.min():.2f}, {bias_scores.max():.2f}]")
    print(f"Mean bias score:  {bias_scores.mean():.2f}")
    print(f"\nBias distribution:")
    print(f"  Progressive:  {n_progressive:,} ({n_progressive/len(bias_scores)*100:.1f}%)")
    print(f"  Neutral:      {n_neutral:,} ({n_neutral/len(bias_scores)*100:.1f}%)")
    print(f"  Conservative: {n_conservative:,} ({n_conservative/len(bias_scores)*100:.1f}%)")
    print(f"\nSaved files:")
    print(f"  {scores_path}")
    print(f"  {csv_path}")
    print("="*80)

    return bias_scores, df


def verify_predictions(csv_path='data/processed/all_articles_with_bias.csv',
                       n_samples=10):
    """
    저장된 예측 결과 샘플 검증

    Args:
        csv_path: 예측 결과 CSV 경로
        n_samples: 출력할 샘플 수
    """

    print("\n" + "="*80)
    print("PREDICTION VERIFICATION (Sample)")
    print("="*80)

    if not os.path.exists(csv_path):
        print(f"  File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"\nLoaded: {len(df):,} articles")
    print(f"Columns: {df.columns.tolist()}")

    # 랜덤 샘플 출력
    print(f"\nRandom {n_samples} samples:")
    sample = df.sample(n_samples, random_state=42)

    for _, row in sample.iterrows():
        initial = row['bias_initial']
        predicted = row['bias_predicted']
        diff = abs(predicted - initial)

        print(f"\n  Title:     {str(row['제목'])[:55]}...")
        print(f"  Media:     {row['언론사']}")
        print(f"  Initial:   {initial:+.1f}")
        print(f"  Predicted: {predicted:+.1f}")
        print(f"  Diff:      {diff:.1f}")


if __name__ == '__main__':
    # 전체 기사 편향도 예측
    bias_scores, df = predict_all_bias()

    # 저장된 결과 샘플 검증
    verify_predictions(n_samples=10)

    print("\nDone")