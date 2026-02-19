# 최종 추천 시스템 통합 클래스

# 검증된 최적 설정을 하나의 클래스로 통합
# 간단한 API로 추천 실행 가능

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pickle
from src.recommendation.candidate_retrieval import CandidateRetrieval
from src.recommendation.mmr_reranking import MMRReranker
from src.recommendation.diversity_metrics import evaluate_recommendation


class FinalRecommender:
    """
    최종 추천 시스템
    
    검증된 최적 설정:
      - Retrieval: Balanced-coverage (진영별 쿼터 20%)
      - Reranking: Coverage-MMR (진영별 최소 1개 보장)
      - Lambda: 0.5 (관련성 vs 다양성 균형)
      - Text weight: 0.6 (텍스트 vs 그래프 임베딩)
    
    사용법:
        recommender = FinalRecommender()
        result = recommender.recommend(query_idx=100, n=5)
    """
    
    def __init__(self,
                 text_emb_path='data/embeddings/text_embeddings.npy',
                 graph_emb_path='data/embeddings/graph_embeddings.npy',
                 bias_scores_path='data/embeddings/bias_scores.npy',
                 df_path='data/processed/all_articles_with_bias.csv',
                 lambda_param=0.5,
                 text_weight=0.6,
                 max_per_media=3,
                 dedup_threshold=0.99):
        """
        최종 추천 시스템 초기화
        
        Args:
            text_emb_path:    텍스트 임베딩 경로
            graph_emb_path:   그래프 임베딩 경로
            bias_scores_path: 편향도 점수 경로
            df_path:          기사 DataFrame 경로
            lambda_param:     MMR λ 파라미터 (0~1)
            text_weight:      텍스트 vs 그래프 가중치
            max_per_media:    동일 언론사 최대 기사 수
            dedup_threshold:  중복 판단 임계값
        """
        
        print("="*80)
        print("Initializing Final Recommender System")
        print("="*80)
        
        # 데이터 로드
        print("\nLoading data...")
        self.text_emb    = np.load(text_emb_path)
        self.graph_emb   = np.load(graph_emb_path)
        self.bias_scores = np.load(bias_scores_path)
        self.df          = pd.read_csv(df_path, encoding='utf-8-sig')
        
        print(f"  - Articles:      {len(self.df):,}")
        print(f"  - Text emb dim:  {self.text_emb.shape[1]}")
        print(f"  - Graph emb dim: {self.graph_emb.shape[1]}")
        
        # 하이퍼파라미터 저장
        self.lambda_param     = lambda_param
        self.text_weight      = text_weight
        self.max_per_media    = max_per_media
        self.dedup_threshold  = dedup_threshold
        
        # 컴포넌트 초기화
        print("\nInitializing components...")
        self.retrieval = CandidateRetrieval(
            text_embeddings=self.text_emb,
            graph_embeddings=self.graph_emb,
            df=self.df,
            text_weight=self.text_weight,
            max_per_media=self.max_per_media,
            dedup_threshold=self.dedup_threshold
        )
        
        self.reranker = MMRReranker(
            bias_scores=self.bias_scores,
            lambda_param=self.lambda_param
        )
        
        print("\nConfiguration:")
        print(f"  - Lambda:         {self.lambda_param}")
        print(f"  - Text weight:    {self.text_weight}")
        print(f"  - Max per media:  {self.max_per_media}")
        print(f"  - Dedup thresh:   {self.dedup_threshold}")
        print("\n✓ Recommender ready")
    
    def recommend(self, query_idx, k=30, n=5, return_metrics=False):
        """
        추천 실행
        
        Args:
            query_idx:      쿼리 기사 인덱스
            k:              후보 검색 수
            n:              최종 추천 수
            return_metrics: 다양성 지표 포함 여부
        
        Returns:
            result: 추천 결과 딕셔너리
                {
                    'query': {...},
                    'recommendations': [{...}, ...],
                    'metrics': {...}  (if return_metrics=True)
                }
        """
        
        # Step 1: 후보 검색 (Balanced-coverage)
        candidates, rel_scores = self.retrieval.retrieve_with_balanced_coverage(
            query_idx, self.bias_scores, k=k,
            min_prog_ratio=0.2, min_neut_ratio=0.2, min_cons_ratio=0.2
        )
        
        # Step 2: MMR 리랭킹 (Coverage-MMR)
        rec_indices, mmr_scores = self.reranker.rerank_with_coverage(
            candidates, rel_scores, n=n
        )
        
        # Step 3: 결과 포맷팅
        query_article = self.df.iloc[query_idx]
        
        query_info = {
            'index':      int(query_idx),
            'news_id':    str(query_article.get('news_id', '')),
            'title':      str(query_article['제목']),
            'media':      str(query_article['언론사']),
            'bias':       float(self.bias_scores[query_idx]),
            'url':        str(query_article.get('네이버링크', ''))
        }
        
        rec_articles = []
        sel_rel_scores = []
        
        for idx, mmr_score in zip(rec_indices, mmr_scores):
            article = self.df.iloc[idx]
            # 원래 후보 유사도 조회
            rel_score = rel_scores[list(candidates).index(idx)]
            sel_rel_scores.append(rel_score)
            
            rec_articles.append({
                'index':      int(idx),
                'news_id':    str(article.get('news_id', '')),
                'title':      str(article['제목']),
                'media':      str(article['언론사']),
                'bias':       float(self.bias_scores[idx]),
                'url':        str(article.get('네이버링크', '')),
                'relevance':  float(rel_score),
                'mmr_score':  float(mmr_score)
            })
        
        result = {
            'query':           query_info,
            'recommendations': rec_articles
        }
        
        # Step 4: 다양성 지표 (선택사항)
        if return_metrics:
            metrics = evaluate_recommendation(
                rec_indices, self.bias_scores, np.array(sel_rel_scores)
            )
            # float 변환 (JSON 호환)
            result['metrics'] = {k: float(v) for k, v in metrics.items()}
        
        return result
    
    def recommend_batch(self, query_indices, k=30, n=5):
        """
        다중 쿼리 배치 추천
        
        Args:
            query_indices: 쿼리 인덱스 목록
            k:             후보 검색 수
            n:             최종 추천 수
        
        Returns:
            results: 추천 결과 리스트
        """
        
        results = []
        for q_idx in query_indices:
            result = self.recommend(q_idx, k=k, n=n, return_metrics=False)
            results.append(result)
        return results
    
    def save_config(self, path='data/models/final_recommender_config.pkl'):
        """
        설정 저장
        
        Args:
            path: 저장 경로
        """
        
        config = {
            'lambda_param':    self.lambda_param,
            'text_weight':     self.text_weight,
            'max_per_media':   self.max_per_media,
            'dedup_threshold': self.dedup_threshold,
            'n_articles':      len(self.df)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"\n✓ Configuration saved: {path}")
        return config
    
    @staticmethod
    def load_config(path='data/models/final_recommender_config.pkl'):
        """
        저장된 설정 로드
        
        Args:
            path: 저장 경로
        
        Returns:
            config: 설정 딕셔너리
        """
        
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        print(f"✓ Configuration loaded: {path}")
        return config


# ──────────────────────────────────────────────
# 테스트 및 시연
# ──────────────────────────────────────────────

def demo_recommender():
    """최종 추천 시스템 데모"""
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDER DEMO")
    print("="*80)
    
    # 추천 시스템 초기화
    recommender = FinalRecommender()
    
    # 샘플 쿼리 선택
    np.random.seed(42)
    sample_queries = np.random.choice(len(recommender.df), 3, replace=False)
    
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    for i, query_idx in enumerate(sample_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Example {i}")
        print(f"{'─'*80}")
        
        result = recommender.recommend(query_idx, n=5, return_metrics=True)
        
        # 쿼리 기사
        query = result['query']
        print(f"\n[Query Article]")
        print(f"  Title: {query['title'][:70]}...")
        print(f"  Media: {query['media']}")
        print(f"  Bias:  {query['bias']:+.2f}")
        
        # 추천 결과
        print(f"\n[Recommendations]")
        print(f"  {'#':<3} {'Bias':>7} {'Rel':>7} {'Media':<20} {'Title'}")
        print(f"  " + "-"*75)
        
        for j, rec in enumerate(result['recommendations'], 1):
            print(f"  {j:<3} {rec['bias']:>+7.2f} {rec['relevance']:>7.4f} "
                  f"{rec['media']:<20} {rec['title'][:35]}...")
        
        # 다양성 지표
        metrics = result['metrics']
        print(f"\n[Diversity Metrics]")
        print(f"  ILD:      {metrics['ILD']:.2f}")
        print(f"  Coverage: {metrics['Coverage']:.2f}")
        print(f"  RR-ILD:   {metrics['RR-ILD']:.4f}")
    
    # 설정 저장
    print("\n" + "="*80)
    print("SAVING CONFIGURATION")
    print("="*80)
    config = recommender.save_config()
    
    print("\nSaved configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return recommender


if __name__ == '__main__':
    print("="*80)
    print("Final Recommender System")
    print("="*80)
    
    recommender = demo_recommender()
    
    print("\n" + "="*80)
    print("SYSTEM READY FOR PRODUCTION")
    print("="*80)
    print(f"\nUsage:")
    print(f"  from src.recommendation.final_recommender import FinalRecommender")
    print(f"  recommender = FinalRecommender()")
    print(f"  result = recommender.recommend(query_idx=100, n=5)")
    
    print("\nDone")