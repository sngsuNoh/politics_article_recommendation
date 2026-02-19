# KoBERT based 텍스트 임베딩 추출

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class TextEncoder:
    def __init__(self, model_name="beomi/kcbert-base"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # 모델의 최대 시퀀스 길이 확인
        self.max_length = 300  # 300까지 지원
        print(f"Device: {self.device}")
        print(f"Max sequence length: {self.max_length}")
    
    def encode(self, texts, batch_size=8):
        """텍스트 리스트 → 768-dim 임베딩"""
        embeddings = []
        
        # NaN 값 처리
        texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i+batch_size]
            
            try:
                # 토크나이징
                inputs = self.tokenizer(
                    batch, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length  # 300으로 제한
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 임베딩 추출
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # [CLS] 토큰 사용
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_emb)
                    
            except Exception as e:
                print(f"\nError at batch {i//batch_size}: {e}")
                print(f"Batch size: {len(batch)}")
                print(f"First text length: {len(batch[0]) if batch else 0}")
                raise
        
        return np.vstack(embeddings)

def extract_and_save():
    print("Loading data...")
    df = pd.read_csv('data/processed/all_articles_labeled.csv', encoding='utf-8-sig')
    
    print(f"Total articles: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 본문 데이터 확인
    print(f"\nChecking '본문' column...")
    print(f"Non-null count: {df['본문'].notna().sum()}")
    print(f"Null count: {df['본문'].isna().sum()}")
    
    # 임베딩 추출
    encoder = TextEncoder()
    print("\nExtracting text embeddings...")
    text_embeddings = encoder.encode(df['본문'].tolist(), batch_size=8)
    
    # 저장
    output_path = 'data/embeddings/text_embeddings.npy'
    np.save(output_path, text_embeddings)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {text_embeddings.shape}")
    print(f"Size: {text_embeddings.nbytes / 1024 / 1024:.2f} MB")
    
if __name__ == '__main__':
    extract_and_save()