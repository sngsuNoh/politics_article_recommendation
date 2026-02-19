# 전체 데이터 통합, ID 매핑 생성 

import pandas as pd
import glob
import pickle
from pathlib import Path

def load_all_processed_data(data_dir='data/processed'):
    # 데이터 전부 로드
    csv_files = sorted(glob.glob(f'{data_dir}/naver_politics_*.csv'))
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, encoding='utf-8-sig')
        df_list.append(df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # news_id 기준 중복제거
    full_df = full_df.drop_duplicates(subset='news_id', keep='first')
    
    return full_df

def create_id_mapping(df, save_path='data/embeddings/id_mapping.pkl'):
    # news_id → index 매핑
    id_mapping = {row['news_id']: idx for idx, row in df.iterrows()}
    
    with open(save_path, 'wb') as f:
        pickle.dump(id_mapping, f)
    
    print(f"ID mapping created: {len(id_mapping)} articles")
    return id_mapping

if __name__ == '__main__':
    df = load_all_processed_data()
    print(f"Total articles loaded: {len(df)}")
    
    id_mapping = create_id_mapping(df)