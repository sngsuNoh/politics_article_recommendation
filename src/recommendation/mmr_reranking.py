# 편향 다양성을 고려한 MMR(Maximal Marginal Relevance) 리랭킹 모듈

# 후보 검색이 반환한 Top-K 후보 중에서
# 관련성(relevance)과 편향 다양성(bias diversity)을 동시에 고려하여 최종 N개 선택

# MMR 점수 = λ * relevance + (1 - λ) * bias_diversity

import numpy as np
import pandas as pd


class MMRReranker:

    def __init__(self, bias_scores, lambda_param=0.5):
        """
        Args:
            bias_scores:  전체 기사 편향도 배열 (N,)
            lambda_param: 관련성 vs 다양성 균형 파라미터 (0~1)
                          1.0 = 관련성만, 0.0 = 다양성만
        """

        self.bias_scores  = bias_scores
        self.lambda_param = lambda_param

    def rerank(self, candidates, relevance_scores, n=5):
        """
        MMR 기반 리랭킹

        매 선택 단계에서 아직 선택되지 않은 후보 중
        MMR 점수가 가장 높은 기사를 순서대로 선택
        편향 다양성은 이미 선택된 기사들과의 편향도 차이 중 최솟값(min)으로 측정
        → 이미 선택된 기사 중 가장 가까운 편향과의 거리를 최대화

        Args:
            candidates:       후보 인덱스 배열 (K,)
            relevance_scores: 후보별 관련성 점수 (K,)  후보 검색 유사도 점수
            n:                최종 추천 기사 수

        Returns:
            selected:     선택된 기사 인덱스 배열 (n,)
            mmr_scores:   선택된 기사의 MMR 점수 (n,)
        """

        remaining = list(candidates)
        rel_dict  = dict(zip(candidates, relevance_scores))

        selected    = []
        mmr_scores  = []

        while len(selected) < n and remaining:
            best_idx   = None
            best_score = -np.inf

            for cand in remaining:
                relevance = rel_dict[cand]

                if not selected:
                    # 첫 번째는 관련성만으로 선택
                    bias_div = 1.0
                else:
                    # 이미 선택된 기사들과의 편향도 차이 중 최솟값
                    # /100 으로 정규화하여 relevance 스케일과 맞춤
                    min_bias_diff = min(
                        abs(self.bias_scores[cand] - self.bias_scores[s]) / 100
                        for s in selected
                    )
                    bias_div = min_bias_diff

                score = (self.lambda_param * relevance +
                         (1 - self.lambda_param) * bias_div)

                if score > best_score:
                    best_score = score
                    best_idx   = cand

            selected.append(best_idx)
            mmr_scores.append(best_score)
            remaining.remove(best_idx)

        return np.array(selected), np.array(mmr_scores)

    def rerank_with_coverage(self, candidates, relevance_scores, n=5):
        """
        진보 / 중립 / 보수 Coverage를 보장하는 MMR 리랭킹

        Phase 1: 진보(-30 미만) / 중립(-30~30) / 보수(30 초과) 각 진영에서
                 관련성이 가장 높은 기사를 1개씩 강제 선택 (최대 3개)
        Phase 2: 남은 슬롯은 일반 MMR로 채움

        후보에 특정 진영 기사가 없으면 해당 슬롯은 건너뜀

        Args:
            candidates:       후보 인덱스 배열 (K,)
            relevance_scores: 후보별 관련성 점수 (K,)
            n:                최종 추천 기사 수

        Returns:
            selected:    선택된 기사 인덱스 배열 (n,)
            mmr_scores:  선택된 기사의 MMR 점수 (n,)
        """

        remaining = list(candidates)
        rel_dict  = dict(zip(candidates, relevance_scores))

        selected   = []
        mmr_scores = []

        # Phase 1: 진영별 Coverage 강제 선택
        # 기준을 ±10으로 완화하여 사실 보도 성격 기사에서도 Coverage가 작동하도록 함
        progressive  = [c for c in candidates if self.bias_scores[c] < -10]
        neutral      = [c for c in candidates if -10 <= self.bias_scores[c] <= 10]
        conservative = [c for c in candidates if self.bias_scores[c] > 10]

        for group in [progressive, neutral, conservative]:
            if group and len(selected) < n:
                # 해당 진영 후보 중 관련성 최고 기사 선택
                best = max(group, key=lambda x: rel_dict[x])
                selected.append(best)
                mmr_scores.append(rel_dict[best])
                remaining.remove(best)

        # Phase 2: 남은 슬롯은 일반 MMR로 채움
        while len(selected) < n and remaining:
            best_idx   = None
            best_score = -np.inf

            for cand in remaining:
                relevance = rel_dict[cand]
                min_bias_diff = min(
                    abs(self.bias_scores[cand] - self.bias_scores[s]) / 100
                    for s in selected
                )
                score = (self.lambda_param * relevance +
                         (1 - self.lambda_param) * min_bias_diff)

                if score > best_score:
                    best_score = score
                    best_idx   = cand

            selected.append(best_idx)
            mmr_scores.append(best_score)
            remaining.remove(best_idx)

        return np.array(selected), np.array(mmr_scores)


# ──────────────────────────────────────────────
# 검증 함수
# ──────────────────────────────────────────────

def validate_mmr(reranker, retrieval, df, bias_scores,
                 query_indices, k=30, n=5):
    """
    샘플 쿼리에 대해 3가지 검색 방식 + MMR 리랭킹 결과 비교
    1. Plain retrieval
    2. Bias-filter retrieval
    3. Balanced-coverage retrieval (진영별 쿼터)

    Args:
        reranker:      MMRReranker 인스턴스
        retrieval:     CandidateRetrieval 인스턴스
        df:            전체 기사 DataFrame
        bias_scores:   전체 기사 편향도 배열
        query_indices: 검증할 쿼리 인덱스 목록
        k:             후보 검색 수
        n:             최종 추천 수
    """

    print("\n" + "="*80)
    print("MMR RERANKING VALIDATION  (3 retrieval strategies)")
    print("="*80)
    print(f"  lambda = {reranker.lambda_param}  "
          f"(relevance: {reranker.lambda_param:.1f}, "
          f"diversity: {1-reranker.lambda_param:.1f})")

    for q_num, query_idx in enumerate(query_indices, 1):
        query = df.iloc[query_idx]
        query_bias = bias_scores[query_idx]

        print(f"\n{'='*80}")
        print(f"[Query {q_num}]")
        print(f"  Title: {str(query['제목'])[:60]}...")
        print(f"  Media: {query['언론사']}  Bias: {query_bias:+.2f}")

        # 세 가지 검색 방식으로 후보 확보
        cands_plain,  rel_plain  = retrieval.retrieve(query_idx, k=k)
        cands_filter, rel_filter = retrieval.retrieve_with_bias_filter(
            query_idx, bias_scores, k=k, min_opposite_ratio=0.3
        )
        cands_balanced, rel_balanced = retrieval.retrieve_with_balanced_coverage(
            query_idx, bias_scores, k=k,
            min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
        )

        for label, candidates, rel_scores in [
            ("Plain retrieval",       cands_plain,    rel_plain),
            ("Bias-filter retrieval", cands_filter,   rel_filter),
            ("Balanced-coverage",     cands_balanced, rel_balanced),
        ]:
            selected, mmr_sc = reranker.rerank_with_coverage(
                candidates, rel_scores, n=n
            )
            sel_biases = bias_scores[selected]
            n_prog = (sel_biases < -10).sum()
            n_neut = ((sel_biases >= -10) & (sel_biases <= 10)).sum()
            n_cons = (sel_biases > 10).sum()

            print(f"\n  [{label}]")
            print(f"  {'Rank':<5} {'Bias':>7} {'Rel':>7} {'MMR':>7} "
                  f"{'Media':<20} {'Title'}")
            print(f"  " + "-"*80)

            for rank, (idx, mmr_s) in enumerate(zip(selected, mmr_sc), 1):
                art   = df.iloc[idx]
                cand_list = list(candidates)
                rel_s = rel_scores[cand_list.index(idx)] if idx in cand_list else 0.0
                print(f"  {rank:<5} {bias_scores[idx]:>+7.2f} {rel_s:>7.4f} {mmr_s:>7.4f} "
                      f"{art['언론사']:<20} {str(art['제목'])[:28]}...")

            print(f"\n  Bias Std: {sel_biases.std():.2f}  "
                  f"ILD: {_calc_ild(sel_biases):.2f}  "
                  f"[Prog(bias<-10):{n_prog}  "
                  f"Neut(-10~10):{n_neut}  "
                  f"Cons(bias>10):{n_cons}]")


def compare_lambda(retrieval, df, bias_scores, query_idx,
                   lambdas=(0.3, 0.5, 0.7, 1.0), k=30, n=5):
    """
    λ 값에 따른 추천 결과 변화 비교

    Args:
        retrieval:   CandidateRetrieval 인스턴스
        df:          전체 기사 DataFrame
        bias_scores: 전체 기사 편향도 배열
        query_idx:   비교할 쿼리 인덱스
        lambdas:     비교할 λ 값 목록
        k:           후보 검색 수
        n:           최종 추천 수
    """

    print("\n" + "="*80)
    print("LAMBDA COMPARISON")
    print("="*80)

    query = df.iloc[query_idx]
    print(f"\nQuery: {str(query['제목'])[:60]}...")
    print(f"Media: {query['언론사']}  Bias: {bias_scores[query_idx]:+.2f}")

    # 후보는 한 번만 검색 (balanced-coverage 방식 사용)
    candidates, rel_scores = retrieval.retrieve_with_balanced_coverage(
        query_idx, bias_scores, k=k,
        min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
    )

    print(f"\n{'Lambda':<8} {'Bias Std':>9} {'ILD':>7} "
          f"{'Prog':>5} {'Neut':>5} {'Cons':>5}  Selected Media")
    print("-"*80)

    for lam in lambdas:
        reranker = MMRReranker(bias_scores, lambda_param=lam)
        selected, _ = reranker.rerank(candidates, rel_scores, n=n)

        sel_biases = bias_scores[selected]
        bias_std   = sel_biases.std()
        ild        = _calc_ild(sel_biases)
        n_prog     = (sel_biases < -10).sum()
        n_neut     = ((sel_biases >= -10) & (sel_biases <= 10)).sum()
        n_cons     = (sel_biases > 10).sum()
        media_list = ", ".join(df.iloc[selected]['언론사'].tolist())

        print(f"  {lam:<6} {bias_std:>9.2f} {ild:>7.2f} "
              f"{n_prog:>5} {n_neut:>5} {n_cons:>5}  {media_list}")


def compare_rerank_modes(retrieval, df, bias_scores, query_idx,
                         lambda_param=0.5, k=30, n=5):
    """
    기본 MMR vs Coverage 보장 MMR 비교

    Args:
        retrieval:    CandidateRetrieval 인스턴스
        df:           전체 기사 DataFrame
        bias_scores:  전체 기사 편향도 배열
        query_idx:    비교할 쿼리 인덱스
        lambda_param: MMR λ 파라미터
        k:            후보 검색 수
        n:            최종 추천 수
    """

    print("\n" + "="*80)
    print("MMR vs COVERAGE-MMR COMPARISON")
    print("="*80)

    query = df.iloc[query_idx]
    print(f"\nQuery: {str(query['제목'])[:60]}...")
    print(f"Media: {query['언론사']}  Bias: {bias_scores[query_idx]:+.2f}")

    candidates, rel_scores = retrieval.retrieve_with_balanced_coverage(
        query_idx, bias_scores, k=k,
        min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
    )
    reranker = MMRReranker(bias_scores, lambda_param=lambda_param)

    for label, method in [("MMR", reranker.rerank),
                           ("Coverage-MMR", reranker.rerank_with_coverage)]:
        selected, _ = method(candidates, rel_scores, n=n)
        sel_biases  = bias_scores[selected]

        print(f"\n  [{label}]")
        print(f"  {'Rank':<5} {'Bias':>7} {'Media':<20} {'Title'}")
        print(f"  " + "-"*65)

        for rank, idx in enumerate(selected, 1):
            art = df.iloc[idx]
            print(f"  {rank:<5} {bias_scores[idx]:>+7.2f} "
                  f"{art['언론사']:<20} {str(art['제목'])[:30]}...")

        n_prog = (sel_biases < -10).sum()
        n_neut = ((sel_biases >= -10) & (sel_biases <= 10)).sum()
        n_cons = (sel_biases > 10).sum()
        print(f"\n  Bias Std: {sel_biases.std():.2f}  "
              f"ILD: {_calc_ild(sel_biases):.2f}  "
              f"[Prog(bias<-10):{n_prog} "
              f"Neut(-10~10):{n_neut} "
              f"Cons(bias>10):{n_cons}]")


def _calc_ild(biases):
    """
    추천 리스트 내 모든 쌍의 편향도 차이 평균 (ILD 미리보기용)

    Args:
        biases: 편향도 배열

    Returns:
        ild: 평균 편향도 차이
    """
    if len(biases) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(biases)):
        for j in range(i + 1, len(biases)):
            total += abs(biases[i] - biases[j])
            count += 1
    return total / count


if __name__ == '__main__':
    import sys
    import os
    # 프로젝트 루트를 path에 추가
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.recommendation.candidate_retrieval import CandidateRetrieval

    print("="*80)
    print("Task 14: MMR Reranking - Validation")
    print("="*80)

    # 데이터 로드
    print("\nLoading data...")
    import numpy as np
    import pandas as pd

    text_emb    = np.load('data/embeddings/text_embeddings.npy')
    graph_emb   = np.load('data/embeddings/graph_embeddings.npy')
    bias_scores = np.load('data/embeddings/bias_scores.npy')
    df = pd.read_csv('data/processed/all_articles_with_bias.csv',
                     encoding='utf-8-sig')

    print(f"  - Articles: {len(df):,}")

    # 후보 검색기 초기화
    print("\nInitializing retrieval system...")
    retrieval = CandidateRetrieval(
        text_embeddings=text_emb,
        graph_embeddings=graph_emb,
        df=df,
        text_weight=0.6,
        max_per_media=3,
        dedup_threshold=0.99
    )

    # MMR 리랭커 초기화 (lambda=0.5)
    reranker = MMRReranker(bias_scores, lambda_param=0.5)

    # 검증용 쿼리 선택
    np.random.seed(42)
    query_indices = np.random.choice(len(df), 3, replace=False).tolist()

    # Plain vs Bias-filter vs Balanced-coverage 검색 + Coverage-MMR 비교
    validate_mmr(reranker, retrieval, df, bias_scores,
                 query_indices=query_indices, k=30, n=5)

    # λ 값 비교 (balanced-coverage 검색 기반)
    print("\n[Lambda comparison uses balanced-coverage retrieval]")
    compare_lambda(retrieval, df, bias_scores,
                   query_idx=query_indices[0],
                   lambdas=(0.3, 0.5, 0.7, 1.0), k=30, n=5)

    # 기본 MMR vs Coverage-MMR 비교 (balanced-coverage 검색 기반)
    print("\n[Rerank mode comparison uses balanced-coverage retrieval]")
    compare_rerank_modes(retrieval, df, bias_scores,
                         query_idx=query_indices[0],
                         lambda_param=0.5, k=30, n=5)

    print("\nDone")