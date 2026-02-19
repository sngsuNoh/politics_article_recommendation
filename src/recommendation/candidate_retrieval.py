# 쿼리 기사와 유사한 Top-K 후보 기사를 검색하는 후보 검색 모듈
#
# 텍스트 임베딩과 그래프 임베딩의 코사인 유사도를 가중 결합하여
# 전체 기사 중 쿼리 기사와 가장 유사한 Top-K 후보를 반환

import numpy as np
import pandas as pd


class CandidateRetrieval:

    def __init__(self,
                 text_embeddings,
                 graph_embeddings,
                 df,
                 text_weight=0.6,
                 max_per_media=3,
                 dedup_threshold=0.99):
        """
        Args:
            text_embeddings:  텍스트 임베딩 (N, 768)
            graph_embeddings: 그래프 임베딩 (N, 64)
            df:               전체 기사 DataFrame (제목, 언론사 컬럼 필요)
            text_weight:      텍스트 유사도 가중치 (0~1)
            max_per_media:    후보에 포함할 동일 언론사 기사 최대 수
            dedup_threshold:  이 유사도 이상이면 중복으로 판단하여 제외 (0~1)
        """

        self.text_weight     = text_weight
        self.graph_weight    = 1.0 - text_weight
        self.n_articles      = len(text_embeddings)
        self.max_per_media   = max_per_media
        self.dedup_threshold = dedup_threshold
        self.media_arr       = df['언론사'].values

        # 코사인 유사도 계산을 위해 L2 정규화하여 저장
        # 정규화된 벡터 간 내적 = 코사인 유사도
        self.text_emb_norm  = self._normalize(text_embeddings)
        self.graph_emb_norm = self._normalize(graph_embeddings)

        # 방법 A: 제목 완전 일치 중복 인덱스 사전 계산
        # 동일 제목이 여러 행에 존재할 경우 첫 번째만 유지하고 나머지는 제외
        print("Preprocessing duplicate titles...")
        titles = df['제목'].astype(str).values
        _, first_occurrence = np.unique(titles, return_index=True)
        self.dedup_mask = np.zeros(self.n_articles, dtype=bool)
        self.dedup_mask[first_occurrence] = True   # True = 유효한 기사
        n_duped = self.n_articles - self.dedup_mask.sum()
        print(f"  - Duplicate titles removed: {n_duped:,} articles")

        print(f"CandidateRetrieval initialized")
        print(f"  - Articles:        {self.n_articles:,} (valid: {self.dedup_mask.sum():,})")
        print(f"  - Text dim:        {text_embeddings.shape[1]}")
        print(f"  - Graph dim:       {graph_embeddings.shape[1]}")
        print(f"  - Text weight:     {self.text_weight:.2f}")
        print(f"  - Graph weight:    {self.graph_weight:.2f}")
        print(f"  - Max per media:   {self.max_per_media}")
        print(f"  - Dedup threshold: {self.dedup_threshold}")

    @staticmethod
    def _normalize(embeddings):
        """
        L2 정규화 (각 벡터를 단위 벡터로 변환)
        정규화된 벡터 간 내적이 코사인 유사도와 동일해짐

        Args:
            embeddings: (N, D) 임베딩 행렬

        Returns:
            normalized: (N, D) 정규화된 임베딩
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 0 벡터 방지 (norm이 0인 경우 1로 대체)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def retrieve(self, query_idx, k=30):
        """
        단일 쿼리 기사에 대해 Top-K 유사 기사 검색

        텍스트 유사도와 그래프 유사도를 가중 결합하여 최종 유사도 계산
        combined_sim = text_weight * text_sim + graph_weight * graph_sim

        후보 필터링 순서:
          1. 자기 자신 제외
          2. 방법 A: 제목 완전 일치 중복 제외 (사전 계산된 dedup_mask 활용)
          3. 방법 C: 유사도 임계값(dedup_threshold) 초과 기사 제외
          4. 동일 언론사 max_per_media 상한 적용
          5. 잔여 후보 중 유사도 상위 k개 반환

        Args:
            query_idx: 쿼리 기사 인덱스 (0-based)
            k:         반환할 후보 수

        Returns:
            candidates: Top-K 후보 인덱스 배열 (k,)
            scores:     해당 후보들의 유사도 점수 (k,)
        """

        # 텍스트 코사인 유사도: 쿼리 벡터와 전체 벡터의 내적
        text_sim  = self.text_emb_norm  @ self.text_emb_norm[query_idx]

        # 그래프 코사인 유사도
        graph_sim = self.graph_emb_norm @ self.graph_emb_norm[query_idx]

        # 가중 결합
        combined_sim = self.text_weight * text_sim + self.graph_weight * graph_sim

        # 1. 자기 자신 제외
        combined_sim[query_idx] = -np.inf

        # 2. 방법 A: 제목 완전 일치 중복 제외
        combined_sim[~self.dedup_mask] = -np.inf

        # 3. 방법 C: 유사도 임계값 초과 기사 제외 (사실상 동일 기사)
        combined_sim[combined_sim >= self.dedup_threshold] = -np.inf

        # 4. 동일 언론사 상한 적용
        # 유사도 내림차순으로 순회하며 언론사별 카운트가 max_per_media 초과 시 제외
        sorted_indices = np.argsort(combined_sim)[::-1]
        media_count = {}
        candidates  = []

        for idx in sorted_indices:
            if combined_sim[idx] == -np.inf:
                break  # 유효한 후보 소진
            media = self.media_arr[idx]
            count = media_count.get(media, 0)
            if count < self.max_per_media:
                candidates.append(idx)
                media_count[media] = count + 1
            if len(candidates) == k:
                break

        candidates = np.array(candidates)
        scores     = combined_sim[candidates]

        return candidates, scores

    def retrieve_batch(self, query_indices, k=30):
        """
        다중 쿼리 기사에 대한 배치 검색

        Args:
            query_indices: 쿼리 기사 인덱스 목록
            k:             각 쿼리당 반환할 후보 수

        Returns:
            results: [(candidates, scores), ...] 길이 len(query_indices) 리스트
        """

        results = []
        for q_idx in query_indices:
            candidates, scores = self.retrieve(q_idx, k)
            results.append((candidates, scores))
        return results

    def retrieve_with_bias_filter(self, query_idx, bias_scores,
                                  k=30, min_opposite_ratio=0.3):
        """
        반대 편향 기사 비율을 보장하는 후보 검색
        Top-K 안에서 쿼리와 반대 편향 기사가 최소 min_opposite_ratio 이상 포함되도록 조정
        중복 제거 및 언론사 상한은 retrieve()와 동일하게 적용

        Args:
            query_idx:          쿼리 기사 인덱스
            bias_scores:        전체 기사 편향도 배열 (N,)
            k:                  반환할 후보 수
            min_opposite_ratio: 반대 편향 기사 최소 비율 (0~1)

        Returns:
            candidates: 조정된 Top-K 후보 인덱스 배열 (k,)
            scores:     해당 후보들의 유사도 점수 (k,)
        """

        # 전체 유사도 계산
        text_sim  = self.text_emb_norm  @ self.text_emb_norm[query_idx]
        graph_sim = self.graph_emb_norm @ self.graph_emb_norm[query_idx]
        combined_sim = self.text_weight * text_sim + self.graph_weight * graph_sim

        # 기본 필터 적용 (자기 자신, 중복 제목, 유사도 임계값)
        combined_sim[query_idx]          = -np.inf
        combined_sim[~self.dedup_mask]   = -np.inf
        combined_sim[combined_sim >= self.dedup_threshold] = -np.inf

        query_bias = bias_scores[query_idx]
        n_opposite_required = int(k * min_opposite_ratio)

        # 반대 편향 여부 판단 (편향도 차이 50 이상)
        is_opposite = np.abs(bias_scores - query_bias) >= 50
        is_opposite[query_idx] = False

        # 반대 편향 후보 중 언론사 상한 적용하며 상위 n_opposite_required 개 선택
        opposite_sim = combined_sim.copy()
        opposite_sim[~is_opposite] = -np.inf

        opposite_top = []
        media_count  = {}
        for idx in np.argsort(opposite_sim)[::-1]:
            if opposite_sim[idx] == -np.inf:
                break
            media = self.media_arr[idx]
            count = media_count.get(media, 0)
            if count < self.max_per_media:
                opposite_top.append(idx)
                media_count[media] = count + 1
            if len(opposite_top) == n_opposite_required:
                break

        # 나머지는 전체 유사도 기준으로 언론사 상한 적용하며 채움
        opposite_set = set(opposite_top)
        additional   = []
        media_count  = {}
        for idx in np.argsort(combined_sim)[::-1]:
            if combined_sim[idx] == -np.inf:
                break
            if idx in opposite_set:
                continue
            media = self.media_arr[idx]
            count = media_count.get(media, 0)
            if count < self.max_per_media:
                additional.append(idx)
                media_count[media] = count + 1
            if len(additional) == k - len(opposite_top):
                break

        candidates = np.array(opposite_top + additional)
        scores     = combined_sim[candidates]

        # 점수 기준으로 재정렬
        order = np.argsort(scores)[::-1]
        return candidates[order], scores[order]

    def retrieve_with_balanced_coverage(self, query_idx, bias_scores,
                                        k=30,
                                        min_prog_ratio=0.2,
                                        min_neut_ratio=0.2,
                                        min_cons_ratio=0.2,
                                        prog_threshold=-10,
                                        cons_threshold=10):
        """
        진보 / 중립 / 보수 진영별 최소 비율을 보장하는 후보 검색
        
        쿼리 편향과 무관하게 후보 k개 중:
          - 진보(bias < prog_threshold) 기사: 최소 k * min_prog_ratio 개
          - 중립(prog_threshold ≤ bias ≤ cons_threshold): 최소 k * min_neut_ratio 개
          - 보수(bias > cons_threshold): 최소 k * min_cons_ratio 개
        를 포함하도록 구성
        
        각 진영에서 유사도가 높은 순으로 쿼터를 채우고,
        남은 슬롯은 전체 유사도 순으로 채움

        Args:
            query_idx:        쿼리 기사 인덱스
            bias_scores:      전체 기사 편향도 배열 (N,)
            k:                반환할 후보 수
            min_prog_ratio:   진보 기사 최소 비율 (0~1)
            min_neut_ratio:   중립 기사 최소 비율 (0~1)
            min_cons_ratio:   보수 기사 최소 비율 (0~1)
            prog_threshold:   진보 판단 기준 (이 값 미만)
            cons_threshold:   보수 판단 기준 (이 값 초과)

        Returns:
            candidates: 진영별 쿼터가 반영된 Top-K 후보 인덱스 배열 (k,)
            scores:     해당 후보들의 유사도 점수 (k,)
        """

        # 전체 유사도 계산
        text_sim  = self.text_emb_norm  @ self.text_emb_norm[query_idx]
        graph_sim = self.graph_emb_norm @ self.graph_emb_norm[query_idx]
        combined_sim = self.text_weight * text_sim + self.graph_weight * graph_sim

        # 기본 필터 적용
        combined_sim[query_idx]        = -np.inf
        combined_sim[~self.dedup_mask] = -np.inf
        combined_sim[combined_sim >= self.dedup_threshold] = -np.inf

        # 각 진영 필요 개수 계산
        n_prog_required = int(k * min_prog_ratio)
        n_neut_required = int(k * min_neut_ratio)
        n_cons_required = int(k * min_cons_ratio)

        # 진영 분류
        is_prog = bias_scores < prog_threshold
        is_neut = (bias_scores >= prog_threshold) & (bias_scores <= cons_threshold)
        is_cons = bias_scores > cons_threshold

        selected = []
        media_count = {}

        # Phase 1: 각 진영에서 쿼터만큼 선택
        for is_group, n_required in [(is_prog, n_prog_required),
                                      (is_neut, n_neut_required),
                                      (is_cons, n_cons_required)]:
            group_sim = combined_sim.copy()
            group_sim[~is_group] = -np.inf

            group_selected = []
            for idx in np.argsort(group_sim)[::-1]:
                if group_sim[idx] == -np.inf:
                    break
                if idx in selected:
                    continue
                media = self.media_arr[idx]
                count = media_count.get(media, 0)
                if count < self.max_per_media:
                    group_selected.append(idx)
                    media_count[media] = count + 1
                if len(group_selected) == n_required:
                    break

            selected.extend(group_selected)

        # Phase 2: 남은 슬롯은 전체 유사도 순으로 채움
        selected_set = set(selected)
        for idx in np.argsort(combined_sim)[::-1]:
            if combined_sim[idx] == -np.inf:
                break
            if idx in selected_set:
                continue
            media = self.media_arr[idx]
            count = media_count.get(media, 0)
            if count < self.max_per_media:
                selected.append(idx)
                media_count[media] = count + 1
            if len(selected) == k:
                break

        candidates = np.array(selected)
        scores     = combined_sim[candidates]

        # 점수 기준으로 재정렬
        order = np.argsort(scores)[::-1]
        return candidates[order], scores[order]


# ──────────────────────────────────────────────
# 검증 함수
# ──────────────────────────────────────────────

def validate_retrieval(retrieval, df, n_queries=5, k=30):
    """
    샘플 쿼리로 후보 검색 결과를 출력하여 정상 동작 검증

    Args:
        retrieval: CandidateRetrieval 인스턴스
        df:        전체 기사 DataFrame (제목, 언론사, bias_predicted 컬럼 필요)
        n_queries: 검증할 쿼리 수
        k:         검색할 후보 수
    """

    print("\n" + "="*80)
    print("RETRIEVAL VALIDATION")
    print("="*80)

    np.random.seed(42)
    query_indices = np.random.choice(len(df), n_queries, replace=False)

    for q_num, query_idx in enumerate(query_indices, 1):
        query = df.iloc[query_idx]

        print(f"\n[Query {q_num}]")
        print(f"  Index:  {query_idx}")
        print(f"  Title:  {str(query['제목'])[:60]}...")
        print(f"  Media:  {query['언론사']}")
        if 'bias_predicted' in df.columns:
            print(f"  Bias:   {query['bias_predicted']:+.2f}")

        # 후보 검색
        candidates, scores = retrieval.retrieve(query_idx, k=k)

        print(f"\n  Top-5 candidates (out of {k}):")
        print(f"  {'Rank':<5} {'Score':>7} {'Media':<20} {'Title'}")
        print(f"  " + "-"*75)

        for rank, (cand_idx, score) in enumerate(zip(candidates[:5], scores[:5]), 1):
            cand = df.iloc[cand_idx]
            title_preview = str(cand['제목'])[:40]
            print(f"  {rank:<5} {score:>7.4f} {cand['언론사']:<20} {title_preview}...")

        # 유사도 점수 분포
        print(f"\n  Score distribution (top-{k}):")
        print(f"    Max:    {scores[0]:.4f}")
        print(f"    Min:    {scores[-1]:.4f}")
        print(f"    Mean:   {scores.mean():.4f}")

        # 언론사 다양성 확인
        candidate_media = df.iloc[candidates]['언론사'].value_counts()
        print(f"\n  Media diversity (top-{k}): {len(candidate_media)} outlets")
        for media, cnt in candidate_media.head(5).items():
            print(f"    {media}: {cnt}")


def compare_max_per_media(text_emb, graph_emb, df, bias_scores,
                          query_indices, k=30):
    """
    max_per_media=2 vs max_per_media=3 결과 비교

    Args:
        text_emb:      텍스트 임베딩
        graph_emb:     그래프 임베딩
        df:            전체 기사 DataFrame
        bias_scores:   전체 기사 편향도 배열
        query_indices: 비교할 쿼리 인덱스 목록
        k:             검색할 후보 수
    """

    print("\n" + "="*80)
    print("MAX_PER_MEDIA COMPARISON  (2 vs 3)")
    print("="*80)

    retrieval_2 = CandidateRetrieval(
        text_emb, graph_emb, df, text_weight=0.6, max_per_media=2
    )
    retrieval_3 = CandidateRetrieval(
        text_emb, graph_emb, df, text_weight=0.6, max_per_media=3
    )

    print(f"\n{'Query':<6} {'Media':<20} {'Bias':>6} | "
          f"{'Outlets(2)':>10} {'MaxCnt(2)':>9} | "
          f"{'Outlets(3)':>10} {'MaxCnt(3)':>9}")
    print("-"*80)

    for query_idx in query_indices:
        query        = df.iloc[query_idx]
        query_media  = query['언론사']
        query_bias   = bias_scores[query_idx]

        cands_2, _ = retrieval_2.retrieve(query_idx, k=k)
        cands_3, _ = retrieval_3.retrieve(query_idx, k=k)

        media_2 = df.iloc[cands_2]['언론사'].value_counts()
        media_3 = df.iloc[cands_3]['언론사'].value_counts()

        outlets_2  = len(media_2)
        max_cnt_2  = media_2.max() if len(media_2) > 0 else 0
        outlets_3  = len(media_3)
        max_cnt_3  = media_3.max() if len(media_3) > 0 else 0

        print(f"  {query_idx:<6} {query_media:<20} {query_bias:>+6.1f} | "
              f"{outlets_2:>10} {max_cnt_2:>9} | "
              f"{outlets_3:>10} {max_cnt_3:>9}")

    # 각 설정의 상세 후보 출력
    for label, r in [("max_per_media = 2", retrieval_2),
                     ("max_per_media = 3", retrieval_3)]:
        print(f"\n--- {label} ---")
        for query_idx in query_indices:
            query = df.iloc[query_idx]
            cands, _ = r.retrieve(query_idx, k=k)
            media_counts = df.iloc[cands]['언론사'].value_counts()

            print(f"\n  Query: {str(query['제목'])[:50]}...")
            print(f"  Media diversity: {len(media_counts)} outlets  "
                  f"(max {media_counts.max() if len(media_counts) > 0 else 0} per outlet)")
            top_media_str = "  /  ".join(
                [f"{m}:{c}" for m, c in media_counts.head(5).items()]
            )
            print(f"  Top media: {top_media_str}")


def validate_bias_filter(retrieval, df, bias_scores, n_queries=3, k=30):
    """
    편향 필터 적용 검색 결과 검증

    Args:
        retrieval:   CandidateRetrieval 인스턴스
        df:          전체 기사 DataFrame
        bias_scores: 전체 기사 편향도 배열
        n_queries:   검증할 쿼리 수
        k:           검색할 후보 수
    """

    print("\n" + "="*80)
    print("BIAS FILTER RETRIEVAL VALIDATION")
    print("="*80)

    # 편향이 강한 기사를 쿼리로 선택 (보수/진보 각각)
    conservative_idx = np.where(bias_scores > 50)[0]
    progressive_idx  = np.where(bias_scores < -50)[0]

    test_queries = []
    if len(conservative_idx) > 0:
        test_queries.append(('Conservative query', conservative_idx[0]))
    if len(progressive_idx) > 0:
        test_queries.append(('Progressive query',  progressive_idx[0]))

    for label, query_idx in test_queries[:n_queries]:
        query = df.iloc[query_idx]
        query_bias = bias_scores[query_idx]

        print(f"\n[{label}]")
        print(f"  Media: {query['언론사']}  Bias: {query_bias:+.2f}")

        # 필터 없는 기본 검색
        cands_plain, _ = retrieval.retrieve(query_idx, k=k)
        opposite_plain = np.sum(np.abs(bias_scores[cands_plain] - query_bias) >= 50)

        # 필터 적용 검색
        cands_filtered, _ = retrieval.retrieve_with_bias_filter(
            query_idx, bias_scores, k=k, min_opposite_ratio=0.3
        )
        opposite_filtered = np.sum(np.abs(bias_scores[cands_filtered] - query_bias) >= 50)

        print(f"\n  Opposite-bias articles in top-{k}:")
        print(f"    Without filter: {opposite_plain} ({opposite_plain/k*100:.1f}%)")
        print(f"    With filter:    {opposite_filtered} ({opposite_filtered/k*100:.1f}%)")


if __name__ == '__main__':
    print("="*80)
    print("Candidate Retrieval - Validation")
    print("="*80)

    # 데이터 로드
    print("\nLoading embeddings...")
    text_emb  = np.load('data/embeddings/text_embeddings.npy')
    graph_emb = np.load('data/embeddings/graph_embeddings.npy')
    df = pd.read_csv('data/processed/all_articles_with_bias.csv',
                     encoding='utf-8-sig')
    bias_scores = np.load('data/embeddings/bias_scores.npy')

    print(f"  - Text embeddings:  {text_emb.shape}")
    print(f"  - Graph embeddings: {graph_emb.shape}")
    print(f"  - Articles:         {len(df):,}")

    # 후보 검색기 초기화 (기본 max_per_media=3)
    print("\nInitializing retrieval system...")
    retrieval = CandidateRetrieval(
        text_embeddings=text_emb,
        graph_embeddings=graph_emb,
        df=df,
        text_weight=0.6,
        max_per_media=3,
        dedup_threshold=0.99
    )

    # 기본 검색 검증
    np.random.seed(42)
    query_indices = np.random.choice(len(df), 5, replace=False).tolist()
    validate_retrieval(retrieval, df, n_queries=5, k=30)

    # max_per_media 2 vs 3 비교
    compare_max_per_media(text_emb, graph_emb, df, bias_scores,
                          query_indices=query_indices, k=30)

    # 편향 필터 검증
    validate_bias_filter(retrieval, df, bias_scores, n_queries=2, k=30)

    print("\nDone")