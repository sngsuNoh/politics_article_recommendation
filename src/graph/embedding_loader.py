# 그래프 임베딩 로더 (특정 기사의 임베딩 반환)

import numpy as np
import pickle

class GraphEmbeddingLoader:
    def __init__(self, 
                 embedding_path='data/embeddings/graph_embeddings.npy',
                 id_mapping_path='data/embeddings/id_mapping.pkl'):
        
        self.embeddings = np.load(embedding_path)
        
        with open(id_mapping_path, 'rb') as f:
            self.id_mapping = pickle.load(f)
        
        print(f"Loaded graph embeddings: {self.embeddings.shape}")
    
    def get_embedding(self, article_id):
        # 특정 기사 ID의 임베딩 반환
        idx = self.id_mapping.get(article_id)
        if idx is None or idx >= len(self.embeddings):
            return None
        return self.embeddings[idx]
    
    def get_all(self):
        return self.embeddings