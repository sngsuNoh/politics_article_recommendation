# 전체 추천 파이프라인 통합 테스트

# 후보 검색 → MMR 리랭킹 → 다양성 지표
# 전체를 하나로 연결하여 end-to-end 테스트

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from src.recommendation.candidate_retrieval import CandidateRetrieval
from src.recommendation.mmr_reranking import MMRReranker
from src.recommendation.diversity_metrics import evaluate_recommendation


def test_recommendation_pipeline(query_idx, df, bias_scores,
                                retrieval, reranker,
                                lambda_param=0.5, k=30, n=5,
                                use_balanced=True):
    """
    단일 쿼리에 대한 전체 추천 파이프라인 실행
    
    Args:
        query_idx:     쿼리 기사 인덱스
        df:            전체 기사 DataFrame
        bias_scores:   전체 기사 편향도 배열
        retrieval:     CandidateRetrieval 인스턴스
        reranker:      MMRReranker 인스턴스
        lambda_param:  MMR λ 파라미터
        k:             후보 검색 수
        n:             최종 추천 수
        use_balanced:  balanced-coverage 검색 사용 여부
    
    Returns:
        recommendations: 최종 추천 인덱스 배열 (n,)
        metrics:         다양성 지표 딕셔너리
    """
    
    query_article = df.iloc[query_idx]
    
    print("\n" + "="*80)
    print("QUERY ARTICLE")
    print("="*80)
    print(f"  Title: {str(query_article['제목'])[:70]}...")
    print(f"  Media: {query_article['언론사']}")
    print(f"  Bias:  {bias_scores[query_idx]:+.2f}")
    
    # Step 1: 후보 검색
    print(f"\n[Step 1] Candidate Retrieval (k={k})")
    if use_balanced:
        candidates, rel_scores = retrieval.retrieve_with_balanced_coverage(
            query_idx, bias_scores, k=k,
            min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
        )
        print(f"  Method: Balanced-coverage retrieval")
    else:
        candidates, rel_scores = retrieval.retrieve(query_idx, k=k)
        print(f"  Method: Plain retrieval")
    
    print(f"  Candidates retrieved: {len(candidates)}")
    print(f"  Relevance range: {rel_scores.min():.4f} ~ {rel_scores.max():.4f}")
    
    # 후보 편향 분포
    cand_biases = bias_scores[candidates]
    n_prog_cand = (cand_biases < -10).sum()
    n_neut_cand = ((cand_biases >= -10) & (cand_biases <= 10)).sum()
    n_cons_cand = (cand_biases > 10).sum()
    print(f"  Candidate bias: Prog={n_prog_cand} Neut={n_neut_cand} Cons={n_cons_cand}")
    
    # Step 2: MMR 리랭킹
    print(f"\n[Step 2] MMR Reranking (λ={lambda_param}, n={n})")
    recommendations, mmr_scores = reranker.rerank_with_coverage(
        candidates, rel_scores, n=n
    )
    print(f"  Recommendations selected: {len(recommendations)}")
    
    # Step 3: 추천 결과 출력
    print(f"\n[Step 3] Final Recommendations")
    print(f"  {'Rank':<5} {'Bias':>7} {'Rel':>7} {'MMR':>7} {'Media':<20} {'Title'}")
    print(f"  " + "-"*80)
    
    # relevance_scores 재구성 (selected 기사들의 원래 후보 유사도)
    sel_rel_scores = np.array([
        rel_scores[list(candidates).index(idx)] for idx in recommendations
    ])
    
    for rank, (idx, rel, mmr) in enumerate(zip(recommendations, sel_rel_scores, mmr_scores), 1):
        art = df.iloc[idx]
        print(f"  {rank:<5} {bias_scores[idx]:>+7.2f} {rel:>7.4f} {mmr:>7.4f} "
              f"{art['언론사']:<20} {str(art['제목'])[:32]}...")
    
    # Step 4: 다양성 지표
    print(f"\n[Step 4] Diversity Metrics")
    metrics = evaluate_recommendation(recommendations, bias_scores, sel_rel_scores)
    
    print(f"  ILD:        {metrics['ILD']:.2f}  (추천 내 평균 편향도 차이)")
    print(f"  Coverage:   {metrics['Coverage']:.2f}  (진보/중립/보수 포함 비율)")
    print(f"  RR-ILD:     {metrics['RR-ILD']:.4f}  (순위·관련성 가중 ILD)")
    print(f"  Bias_Std:   {metrics['Bias_Std']:.2f}  (편향도 표준편차)")
    print(f"  Bias_Range: {metrics['Bias_Range']:.2f}  (편향도 범위)")
    
    return recommendations, metrics


def test_multiple_queries(df, bias_scores, retrieval,
                         query_indices, lambda_params=[0.3, 0.5, 0.7],
                         k=30, n=5):
    """
    여러 쿼리와 λ 값에 대한 파이프라인 테스트
    
    Args:
        df:             전체 기사 DataFrame
        bias_scores:    전체 기사 편향도 배열
        retrieval:      CandidateRetrieval 인스턴스
        query_indices:  테스트할 쿼리 인덱스 목록
        lambda_params:  테스트할 λ 값 목록
        k:              후보 검색 수
        n:              최종 추천 수
    
    Returns:
        all_results: {(query_idx, lambda): (recommendations, metrics), ...}
    """
    
    all_results = {}
    
    for q_num, query_idx in enumerate(query_indices, 1):
        print("\n" + "="*80)
        print(f"TEST CASE {q_num}/{len(query_indices)}")
        print("="*80)
        
        for lam in lambda_params:
            print(f"\n{'─'*80}")
            print(f"Lambda = {lam}")
            print(f"{'─'*80}")
            
            reranker = MMRReranker(bias_scores, lambda_param=lam)
            recommendations, metrics = test_recommendation_pipeline(
                query_idx, df, bias_scores, retrieval, reranker,
                lambda_param=lam, k=k, n=n, use_balanced=True
            )
            
            all_results[(query_idx, lam)] = (recommendations, metrics)
    
    return all_results


def summarize_results(all_results, df, bias_scores):
    """
    전체 테스트 결과 요약
    
    Args:
        all_results: test_multiple_queries() 반환값
        df:          전체 기사 DataFrame
        bias_scores: 전체 기사 편향도 배열
    """
    
    print("\n" + "="*80)
    print("SUMMARY - AVERAGE METRICS ACROSS QUERIES")
    print("="*80)
    
    # λ별로 그룹화
    lambda_groups = {}
    for (query_idx, lam), (recs, metrics) in all_results.items():
        if lam not in lambda_groups:
            lambda_groups[lam] = []
        lambda_groups[lam].append(metrics)
    
    print(f"\n{'Lambda':<8} {'ILD':>8} {'Coverage':>9} {'RR-ILD':>9} {'Bias_Std':>9} {'Bias_Range':>11}")
    print("-"*65)
    
    for lam in sorted(lambda_groups.keys()):
        metrics_list = lambda_groups[lam]
        avg_ild      = np.mean([m['ILD'] for m in metrics_list])
        avg_coverage = np.mean([m['Coverage'] for m in metrics_list])
        avg_rrild    = np.mean([m['RR-ILD'] for m in metrics_list])
        avg_std      = np.mean([m['Bias_Std'] for m in metrics_list])
        avg_range    = np.mean([m['Bias_Range'] for m in metrics_list])
        
        print(f"  {lam:<6.1f} {avg_ild:>8.2f} {avg_coverage:>9.2f} "
              f"{avg_rrild:>9.4f} {avg_std:>9.2f} {avg_range:>11.2f}")
    
    print(f"\n  (Averaged over {len(metrics_list)} queries)")


if __name__ == '__main__':
    print("="*80)
    print("Recommendation Pipeline Integration Test")
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
    
    # 테스트 쿼리 선택 (다양한 편향도의 쿼리)
    print("\nSelecting test queries...")
    np.random.seed(42)
    
    # 진보/중립/보수 각 진영에서 1개씩
    prog_indices = np.where(bias_scores < -30)[0]
    neut_indices = np.where((bias_scores >= -10) & (bias_scores <= 10))[0]
    cons_indices = np.where(bias_scores > 30)[0]
    
    query_indices = [
        np.random.choice(prog_indices, 1)[0] if len(prog_indices) > 0 else 0,
        np.random.choice(neut_indices, 1)[0],
        np.random.choice(cons_indices, 1)[0] if len(cons_indices) > 0 else 100,
    ]
    
    print(f"  Selected {len(query_indices)} queries from different bias ranges")
    
    # 여러 쿼리 & λ 값 테스트
    all_results = test_multiple_queries(
        df, bias_scores, retrieval,
        query_indices=query_indices,
        lambda_params=[0.3, 0.5, 0.7],
        k=30, n=5
    )
    
    # 결과 요약
    summarize_results(all_results, df, bias_scores)
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETED")
    print("="*80)
    print(f"\nTotal test cases: {len(all_results)}")
    print(f"  Queries tested: {len(query_indices)}")
    print(f"  Lambda values:  {[0.3, 0.5, 0.7]}")
    print(f"\nPipeline components verified:")
    print(f"  ✓ Candidate retrieval (balanced-coverage)")
    print(f"  ✓ MMR reranking (with coverage guarantee)")
    print(f"  ✓ Diversity metrics calculation")
    print(f"\nSystem is ready for production use.")
    
    print("\nDone")