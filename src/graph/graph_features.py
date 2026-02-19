# 그래프 feature 추출

import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm

class GraphFeatureExtractor:
    def __init__(self, 
                 adj_list_path='data/graph/knn_adj_list_per_node.csv',
                 bias_scores=None):
        
        self.adj_df = pd.read_csv(adj_list_path)
        self.bias_scores = bias_scores or {}
        
        # neighbors 파싱
        self.adj_df['neighbors'] = self.adj_df['neighbors'].apply(literal_eval)
    
    def extract_node_features(self, node_idx):
        # 단일 노드 feature 추출
        if node_idx >= len(self.adj_df):
            return None
        
        row = self.adj_df.iloc[node_idx]
        neighbors = row['neighbors']
        
        # 이웃 개수
        degree = len(neighbors)
        
        # 평균 가중치
        avg_weight = np.mean([w for _, w in neighbors]) if neighbors else 0
        
        # 이웃 평균 편향도
        neighbor_biases = [self.bias_scores.get(n, 0) for n, _ in neighbors]
        avg_bias = np.mean(neighbor_biases) if neighbor_biases else 0
        
        # 이웃 편향도 표준편차
        std_bias = np.std(neighbor_biases) if len(neighbor_biases) > 1 else 0
        
        # 반대 입장 존재 여부
        current_bias = self.bias_scores.get(node_idx, 0)
        has_opposite = int(any(abs(nb - current_bias) > 50 
                              for nb in neighbor_biases))
        
        return {
            'degree': degree,
            'avg_weight': avg_weight,
            'neighbor_avg_bias': avg_bias,
            'neighbor_std_bias': std_bias,
            'has_opposite': has_opposite
        }
    
    def extract_all_features(self):
        # 전체 노드 Feature 추출
        features = []
        for idx in tqdm(range(len(self.adj_df))):
            feat = self.extract_node_features(idx)
            features.append(feat)
        
        return pd.DataFrame(features)

if __name__ == '__main__':
    # 편향도 로드
    df = pd.read_csv('data/processed/all_articles_labeled.csv')
    bias_dict = {idx: bias for idx, bias in enumerate(df['bias_initial'])}
    
    # Feature 추출
    extractor = GraphFeatureExtractor(bias_scores=bias_dict)
    graph_features = extractor.extract_all_features()
    
    # 저장
    graph_features.to_csv('data/embeddings/graph_features.csv', index=False)
    print(f"Saved: {graph_features.shape}")