# 언론사 레이블 적용

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.bias_scoring.media_bias_map import get_media_bias
from src.utils.data_loader import load_all_processed_data

def apply_media_labels():
    df = load_all_processed_data()
    
    # 편향도 레이블 적용
    df['bias_initial'] = df['언론사'].apply(get_media_bias)
    
    # 저장
    df.to_csv('data/processed/all_articles_labeled.csv', 
              index=False, encoding='utf-8-sig')
    
    print(f"Labeled {len(df)} articles")
    print(f"Bias distribution:\n{df['bias_initial'].value_counts()}")
    
    return df

if __name__ == '__main__':
    df = apply_media_labels()