# 기사 제목/언론사로 검색하여 추천 결과 확인
# 사용자가 기사를 선택하면 해당 기사에 대한 추천 5개를 출력

import sys
import os

# 프로젝트 루트를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from src.recommendation.final_recommender import FinalRecommender


def search_articles(df, keyword=None, media=None, limit=20):
    """
    기사 검색
    
    Args:
        df:      전체 기사 DataFrame
        keyword: 제목 검색 키워드 (부분 일치)
        media:   언론사 이름 (부분 일치)
        limit:   최대 결과 수
    
    Returns:
        검색 결과 DataFrame
    """
    
    mask = pd.Series([True] * len(df))
    
    if keyword:
        mask &= df['제목'].str.contains(keyword, case=False, na=False)
    
    if media:
        mask &= df['언론사'].str.contains(media, case=False, na=False)
    
    results = df[mask].head(limit)
    return results


def display_search_results(results, bias_scores):
    """
    검색 결과 출력
    
    Args:
        results:     검색 결과 DataFrame
        bias_scores: 전체 기사 편향도 배열
    """
    
    if len(results) == 0:
        print("\n검색 결과가 없습니다.")
        return
    
    print(f"\n검색 결과: {len(results)}개")
    print(f"\n{'No':<4} {'Index':<8} {'Bias':>7} {'Media':<20} {'Title'}")
    print("-" * 100)
    
    for i, (idx, row) in enumerate(results.iterrows(), 1):
        title = str(row['제목'])[:50]
        media = str(row['언론사'])[:18]
        bias = bias_scores[idx]
        print(f"{i:<4} {idx:<8} {bias:>+7.2f} {media:<20} {title}...")


def display_recommendations(query_article, result, bias_scores):
    """
    추천 결과 출력
    
    Args:
        query_article: 쿼리 기사 (DataFrame row)
        result:        추천 결과 (FinalRecommender.recommend 반환값)
        bias_scores:   전체 기사 편향도 배열
    """
    
    print("\n" + "="*100)
    print("추천 결과")
    print("="*100)
    
    # 쿼리 기사 정보
    query = result['query']
    print(f"\n[쿼리 기사]")
    print(f"  제목:     {query['title']}")
    print(f"  언론사:   {query['media']}")
    print(f"  편향도:   {query['bias']:+.2f}")
    if query['url']:
        print(f"  링크:     {query['url']}")
    
    # 추천 기사들
    print(f"\n[추천 기사 5개]")
    print(f"  {'#':<3} {'Bias':>7} {'Rel':>7} {'Media':<20} {'Title'}")
    print("  " + "-"*95)
    
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i:<3} {rec['bias']:>+7.2f} {rec['relevance']:>7.4f} "
              f"{rec['media']:<20} {rec['title'][:50]}...")
    
    # 다양성 지표 (있는 경우)
    if 'metrics' in result:
        metrics = result['metrics']
        print(f"\n[다양성 지표]")
        print(f"  ILD:        {metrics['ILD']:.2f}  (추천 내 평균 편향도 차이)")
        print(f"  Coverage:   {metrics['Coverage']:.2f}  (진보/중립/보수 포함 비율)")
        print(f"  RR-ILD:     {metrics['RR-ILD']:.4f}  (순위·관련성 가중 ILD)")
        print(f"  Bias_Std:   {metrics['Bias_Std']:.2f}  (편향도 표준편차)")
        print(f"  Bias_Range: {metrics['Bias_Range']:.2f}  (편향도 범위)")
    
    # 추천 기사 상세 (선택사항)
    print(f"\n[추천 기사 상세]")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n  {i}. {rec['title']}")
        print(f"     언론사: {rec['media']}  |  편향도: {rec['bias']:+.2f}  |  관련성: {rec['relevance']:.4f}")
        if rec['url']:
            print(f"     링크: {rec['url']}")


def interactive_search_and_recommend():
    """
    인터랙티브 검색 및 추천
    """
    
    print("="*100)
    print("기사 검색 및 추천 시스템")
    print("="*100)
    
    # 추천 시스템 초기화
    print("\n추천 시스템 로딩 중...")
    recommender = FinalRecommender()
    df = recommender.df
    bias_scores = recommender.bias_scores
    
    print(f"✓ 로딩 완료 (총 {len(df):,}개 기사)")
    
    while True:
        print("\n" + "-"*100)
        print("기사 검색")
        print("-"*100)
        
        # 검색 조건 입력
        keyword = input("\n제목 키워드 (Enter=전체): ").strip()
        media = input("언론사 이름 (Enter=전체): ").strip()
        
        if not keyword and not media:
            print("\n최소 하나의 검색 조건을 입력해주세요.")
            continue
        
        # 검색 실행
        results = search_articles(df, keyword=keyword or None, media=media or None, limit=20)
        display_search_results(results, bias_scores)
        
        if len(results) == 0:
            retry = input("\n다시 검색하시겠습니까? (y/n): ").strip().lower()
            if retry != 'y':
                break
            continue
        
        # 기사 선택
        while True:
            try:
                choice = input("\n추천을 받을 기사 번호 (1~{}, 0=다시검색, q=종료): ".format(len(results))).strip()
                
                if choice.lower() == 'q':
                    print("\n프로그램을 종료합니다.")
                    return
                
                if choice == '0':
                    break
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(results):
                    # 선택된 기사의 인덱스
                    selected_idx = results.iloc[choice_num - 1].name
                    selected_article = df.loc[selected_idx]
                    
                    # 추천 실행
                    print("\n추천 생성 중...")
                    result = recommender.recommend(
                        query_idx=selected_idx,
                        n=5,
                        return_metrics=True
                    )
                    
                    # 결과 출력
                    display_recommendations(selected_article, result, bias_scores)
                    
                    # 계속 여부
                    cont = input("\n\n다른 기사를 검색하시겠습니까? (y/n): ").strip().lower()
                    if cont != 'y':
                        print("\n프로그램을 종료합니다.")
                        return
                    break
                else:
                    print(f"1~{len(results)} 사이의 숫자를 입력해주세요.")
            
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
            except Exception as e:
                print(f"오류 발생: {e}")


def quick_test():
    """
    빠른 테스트 (특정 기사로 바로 추천)
    """
    
    print("="*100)
    print("추천 시스템 빠른 테스트")
    print("="*100)
    
    # 추천 시스템 초기화
    print("\n추천 시스템 로딩 중...")
    recommender = FinalRecommender()
    df = recommender.df
    bias_scores = recommender.bias_scores
    
    print(f"✓ 로딩 완료 (총 {len(df):,}개 기사)")
    
    # 샘플 기사 선택 (다양한 편향도)
    np.random.seed(42)
    
    prog_idx = np.random.choice(np.where(bias_scores < -30)[0], 1)[0]
    neut_idx = np.random.choice(np.where((bias_scores >= -10) & (bias_scores <= 10))[0], 1)[0]
    cons_idx = np.random.choice(np.where(bias_scores > 30)[0], 1)[0]
    
    test_indices = [prog_idx, neut_idx, cons_idx]
    
    for i, query_idx in enumerate(test_indices, 1):
        print(f"\n{'='*100}")
        print(f"테스트 {i}/3")
        print(f"{'='*100}")
        
        selected_article = df.iloc[query_idx]
        
        print(f"\n쿼리 기사:")
        print(f"  제목:   {selected_article['제목'][:70]}...")
        print(f"  언론사: {selected_article['언론사']}")
        print(f"  편향도: {bias_scores[query_idx]:+.2f}")
        
        # 추천 실행
        result = recommender.recommend(
            query_idx=query_idx,
            n=5,
            return_metrics=True
        )
        
        # 결과 출력
        display_recommendations(selected_article, result, bias_scores)
        
        if i < len(test_indices):
            input("\n[Enter를 눌러 다음 테스트로 이동...]")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='기사 검색 및 추천 시스템')
    parser.add_argument('--test', action='store_true', 
                       help='빠른 테스트 모드 (샘플 기사 3개로 자동 테스트)')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            quick_test()
        else:
            interactive_search_and_recommend()
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()