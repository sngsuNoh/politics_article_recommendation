# 추천 리스트의 편향 다양성을 정량적으로 측정하는 지표 모듈
#
# ILD, Coverage, RR-ILD, Bias Std 등 4가지 지표 제공
# 추천 시스템 성능 평가 및 λ 파라미터 튜닝의 근거로 사용

import numpy as np
from itertools import combinations


def calculate_ILD(recommendations, bias_scores):
    """
    Intra-List Diversity (ILD)
    
    추천 리스트 내 모든 쌍의 편향도 차이 평균
    값이 클수록 다양한 관점의 기사가 포함되어 있음
    
    Args:
        recommendations: 추천 기사 인덱스 배열 (n,)
        bias_scores:     전체 기사 편향도 배열 (N,)
    
    Returns:
        ild: 0~100 사이 값 (편향도 스케일이 -100~+100이므로)
             0   = 모든 기사가 동일한 편향도
             100 = 최대 편향 차이 (예: -100과 +100 조합)
    """
    
    if len(recommendations) < 2:
        return 0.0
    
    biases = bias_scores[recommendations]
    
    # 모든 쌍의 편향도 차이 계산
    pairs = list(combinations(biases, 2))
    distances = [abs(b1 - b2) for b1, b2 in pairs]
    
    return np.mean(distances)


def calculate_coverage(recommendations, bias_scores,
                      threshold_prog=-10, threshold_cons=10):
    """
    Coverage (진영 커버리지)
    
    진보 / 중립 / 보수 3개 진영이 추천 리스트에 각각 포함되어 있는지 측정
    
    Args:
        recommendations:  추천 기사 인덱스 배열 (n,)
        bias_scores:      전체 기사 편향도 배열 (N,)
        threshold_prog:   진보 판단 기준 (이 값 미만)
        threshold_cons:   보수 판단 기준 (이 값 초과)
    
    Returns:
        coverage: 0.0 ~ 1.0
                  0.0  = 한 진영만 포함
                  0.33 = 한 진영만 포함
                  0.67 = 두 진영 포함
                  1.0  = 세 진영 모두 포함
    """
    
    biases = bias_scores[recommendations]
    
    has_prog = np.any(biases < threshold_prog)
    has_neut = np.any((biases >= threshold_prog) & (biases <= threshold_cons))
    has_cons = np.any(biases > threshold_cons)
    
    count = sum([has_prog, has_neut, has_cons])
    return count / 3.0


def calculate_RR_ILD(recommendations, bias_scores, relevance_scores):
    """
    Rank & Relevance-sensitive ILD (RR-ILD)
    
    상위 순위 + 높은 관련성을 가진 기사 쌍의 다양성을 더 중시
    단순 ILD는 모든 쌍을 동등하게 취급하지만,
    RR-ILD는 1~2위 기사의 다양성이 4~5위보다 더 중요하다고 가정
    
    Args:
        recommendations:   추천 기사 인덱스 배열 (n,)
        bias_scores:       전체 기사 편향도 배열 (N,)
        relevance_scores:  추천 기사의 관련성 점수 배열 (n,)
                           후보 검색에서 반환된 유사도 점수
    
    Returns:
        rr_ild: 가중 다양성 점수
                순위가 높고 관련성이 높은 쌍일수록 더 큰 가중치 부여
    """
    
    n = len(recommendations)
    if n < 2:
        return 0.0
    
    total = 0.0
    weight_sum = 0.0
    
    for i, j in combinations(range(n), 2):
        #    순위 가중치 (상위 순위일수록 높음)
        #    rank_weight = (n - i) * (n - j)
        #    예: n=5일 때 (1,2)쌍 = 5*4=20, (4,5)쌍 = 2*1=2
        rank_weight = (n - i) * (n - j)
        
        #    관련성 가중치 (두 기사의 평균 관련성)
        rel_weight = (relevance_scores[i] + relevance_scores[j]) / 2.0
        
        #    편향도 거리 (0~1로 정규화)
        bias_dist = abs(
            bias_scores[recommendations[i]] - 
            bias_scores[recommendations[j]]
        ) / 100.0
        
        # 가중합
        weight = rank_weight * rel_weight
        total += weight * bias_dist
        weight_sum += weight
    
    # 정규화하여 반환
    return total / weight_sum if weight_sum > 0 else 0.0


def calculate_bias_std(recommendations, bias_scores):
    """
    편향도 표준편차 (Bias Standard Deviation)
    
    추천 리스트 내 편향도 분산을 측정
    ILD와 유사하지만 표준편차는 outlier에 민감
    
    Args:
        recommendations: 추천 기사 인덱스 배열 (n,)
        bias_scores:     전체 기사 편향도 배열 (N,)
    
    Returns:
        std: 표준편차 (0~)
             0    = 모든 기사가 동일한 편향도
             큰 값 = 편향도가 넓게 퍼져 있음
    """
    
    biases = bias_scores[recommendations]
    return np.std(biases)


def calculate_bias_range(recommendations, bias_scores):
    """
    편향도 범위 (Bias Range)
    
    추천 리스트 내 최대 편향도 - 최소 편향도
    
    Args:
        recommendations: 추천 기사 인덱스 배열 (n,)
        bias_scores:     전체 기사 편향도 배열 (N,)
    
    Returns:
        range: 편향도 범위 (0~200)
               0   = 모든 기사가 동일한 편향도
               200 = 최대 범위 (-100 ~ +100)
    """
    
    biases = bias_scores[recommendations]
    return biases.max() - biases.min()


# ──────────────────────────────────────────────
# 종합 평가
# ──────────────────────────────────────────────

def evaluate_recommendation(recommendations, bias_scores, relevance_scores,
                           threshold_prog=-10, threshold_cons=10):
    """
    추천 결과 종합 평가 (4가지 다양성 지표)
    
    Args:
        recommendations:   추천 기사 인덱스 배열 (n,)
        bias_scores:       전체 기사 편향도 배열 (N,)
        relevance_scores:  추천 기사의 관련성 점수 배열 (n,)
        threshold_prog:    진보 판단 기준
        threshold_cons:    보수 판단 기준
    
    Returns:
        metrics: 다양성 지표 딕셔너리
    """
    
    metrics = {
        'ILD':        calculate_ILD(recommendations, bias_scores),
        'Coverage':   calculate_coverage(recommendations, bias_scores,
                                        threshold_prog, threshold_cons),
        'RR-ILD':     calculate_RR_ILD(recommendations, bias_scores, relevance_scores),
        'Bias_Std':   calculate_bias_std(recommendations, bias_scores),
        'Bias_Range': calculate_bias_range(recommendations, bias_scores),
    }
    
    return metrics


def compare_recommendations(rec_dict, bias_scores):
    """
    여러 추천 결과를 나란히 비교
    
    Args:
        rec_dict:    {label: (recommendations, relevance_scores), ...}
        bias_scores: 전체 기사 편향도 배열
    
    Returns:
        comparison_df: 비교 결과 DataFrame
    """
    
    import pandas as pd
    
    results = []
    for label, (recommendations, relevance_scores) in rec_dict.items():
        metrics = evaluate_recommendation(recommendations, bias_scores, relevance_scores)
        metrics['Method'] = label
        results.append(metrics)
    
    df = pd.DataFrame(results)
    # Method 컬럼을 맨 앞으로
    cols = ['Method'] + [c for c in df.columns if c != 'Method']
    return df[cols]


# ──────────────────────────────────────────────
# 검증 및 시연
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.recommendation.candidate_retrieval import CandidateRetrieval
    from src.recommendation.mmr_reranking import MMRReranker
    import pandas as pd
    
    print("="*80)
    print("Diversity Metrics - Validation")
    print("="*80)
    
    # 데이터 로드
    print("\nLoading data...")
    text_emb    = np.load('data/embeddings/text_embeddings.npy')
    graph_emb   = np.load('data/embeddings/graph_embeddings.npy')
    bias_scores = np.load('data/embeddings/bias_scores.npy')
    df = pd.read_csv('data/processed/all_articles_with_bias.csv',
                     encoding='utf-8-sig')
    
    print(f"  - Articles: {len(df):,}")
    
    # 추천 시스템 초기화
    print("\nInitializing recommendation system...")
    retrieval = CandidateRetrieval(
        text_embeddings=text_emb,
        graph_embeddings=graph_emb,
        df=df,
        text_weight=0.6,
        max_per_media=3,
        dedup_threshold=0.99
    )
    
    # 샘플 쿼리 선택
    np.random.seed(42)
    query_idx = np.random.choice(len(df), 1)[0]
    query = df.iloc[query_idx]
    
    print(f"\nQuery article:")
    print(f"  Title: {str(query['제목'])[:60]}...")
    print(f"  Media: {query['언론사']}")
    print(f"  Bias:  {bias_scores[query_idx]:+.2f}")
    
    # 후보 검색 (balanced-coverage)
    candidates, rel_scores = retrieval.retrieve_with_balanced_coverage(
        query_idx, bias_scores, k=30,
        min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
    )
    
    # 여러 λ 값으로 추천 생성 및 지표 비교
    print("\n" + "="*80)
    print("DIVERSITY METRICS COMPARISON  (λ variation)")
    print("="*80)
    
    rec_dict = {}
    for lam in [0.3, 0.5, 0.7, 1.0]:
        reranker = MMRReranker(bias_scores, lambda_param=lam)
        selected, mmr_scores = reranker.rerank_with_coverage(
            candidates, rel_scores, n=5
        )
        # relevance_scores는 selected 기사들의 원래 후보 유사도
        sel_rel_scores = np.array([
            rel_scores[list(candidates).index(idx)] for idx in selected
        ])
        rec_dict[f'λ={lam}'] = (selected, sel_rel_scores)
    
    comparison_df = compare_recommendations(rec_dict, bias_scores)
    
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # 개별 추천 결과 상세 출력
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATIONS")
    print("="*80)
    
    for label, (recommendations, relevance_scores) in rec_dict.items():
        print(f"\n[{label}]")
        print(f"  {'Rank':<5} {'Bias':>7} {'Rel':>7} {'Media':<20} {'Title'}")
        print(f"  " + "-"*70)
        
        for rank, (idx, rel) in enumerate(zip(recommendations, relevance_scores), 1):
            art = df.iloc[idx]
            print(f"  {rank:<5} {bias_scores[idx]:>+7.2f} {rel:>7.4f} "
                  f"{art['언론사']:<20} {str(art['제목'])[:30]}...")
        
        metrics = evaluate_recommendation(recommendations, bias_scores, relevance_scores)
        print(f"\n  Metrics: ILD={metrics['ILD']:.2f}  "
              f"Coverage={metrics['Coverage']:.2f}  "
              f"RR-ILD={metrics['RR-ILD']:.4f}  "
              f"Std={metrics['Bias_Std']:.2f}")
    
    # 지표 해석 가이드
    print("\n" + "="*80)
    print("METRIC INTERPRETATION GUIDE")
    print("="*80)
    print("""
  ILD (Intra-List Diversity):
    - 추천 내 모든 쌍의 평균 편향도 차이
    - 높을수록 다양한 관점 포함
    - 예: 5개 추천에서 ILD=40 → 평균적으로 편향도 차이 40
    
  Coverage:
    - 진보/중립/보수 3개 진영 포함 비율
    - 1.0 = 세 진영 모두 포함 (이상적)
    - 0.67 = 두 진영만 포함
    - 0.33 = 한 진영만 포함
    
  RR-ILD (Rank & Relevance-sensitive ILD):
    - 상위 순위 + 높은 관련성 쌍의 다양성을 더 중시
    - ILD보다 사용자 경험에 가까운 지표
    - 0.0~1.0 범위 (높을수록 좋음)
    
  Bias_Std:
    - 편향도 표준편차
    - 높을수록 편향도가 넓게 분포
    - outlier에 민감
    
  일반적으로 λ가 작을수록 (다양성 중시) ILD, Coverage, RR-ILD가 증가
    """)
    
    print("\nDone")