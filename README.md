# 🗞️ 정치/사회적 편향을 고려한 그래프 기반 다관점 뉴스 추천 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


---

## 👥 Team Members

| 이름 | GitHub | 주요 담당 |
|:---:|:---:|---|
| **노승수** | [@sngsuNoh](https://github.com/sngsuNoh) | 텍스트 임베딩, 편향도 예측 모델, 추천시스템, 평가 |
| **박신지** | [@shinjipark22](https://github.com/shinjipark22) | Airflow 파이프라인, 편향도 라벨링 알고리즘 |
| **이수인** | [@PassionChicken-Leesuin](https://github.com/PassionChicken-Leesuin) | 토픽 기반 라벨링, 그래프 임베딩, Edge list |
| **임은석** | [@yim3001](https://github.com/yim3001) | 데이터 수집, 전처리, BigQuery DB 구성 |

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [핵심 특징](#-핵심-특징)
- [시스템 아키텍처](#️-시스템-아키텍처)
- [기술 스택](#️-기술-스택)
- [디렉토리 구조](#-디렉토리-구조)
- [설치 및 실행](#-설치-및-실행)
- [추천 알고리즘](#-추천-알고리즘)
- [평가 지표](#-평가-지표)
- [실험 결과](#-실험-결과)
- [참고 문헌](#-참고-문헌)
- [License](#-License)

---

## 🎯 프로젝트 개요

### 프로젝트 목적

본 프로젝트는 **다양한 정치적 관점을 균형있게 제공하는 뉴스 추천 시스템**을 개발하여, 사용자가 한 사건에 대한 여러 관점을 접할 수 있도록 지원합니다. 

---

## ✨ 핵심 특징

### 1. 언론사 편향도 예측 모델
- **kcbert-base** 기반 텍스트 임베딩 (768-dim)
- **그래프 임베딩** (64-dim)으로 기사 간 유사도 관계 학습
- **그래프 Feature** (5-dim): 이웃 편향도, 편향도 표준편차, 연결 차수 등
- **XGBoost**: -100(진보) ~ +100(보수) 편향도 예측

### 2. 다관점 추천 알고리즘
- **Balanced-Coverage Retrieval**: 진보/중립/보수 각 진영 20% 쿼터 보장
- **Coverage-MMR Reranking**: 진영별 최소 1개씩 강제 선택 후 MMR 적용

### 3. 검증된 성능
- **Coverage 1.00**: 모든 추천에서 진보/중립/보수 포함
- **평균 ILD 21.96**: 추천 내 편향도 차이 충분
- **평균 RR-ILD 0.22**: 순위와 관련성을 고려한 다양성
- **1000개 쿼리 평가**: 대규모 추천 평가에서 목표 지표 초과 달성

---

## 🏗️ 시스템 아키텍처

```
                   [네이버 뉴스 - 정치 섹션]
                              |
                            크롤링
                              |
                              v
                    [전처리 & 언론사 레이블]
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              [텍스트 임베딩]    [그래프 구조 생성]
               (kcbert-base)            |
                    |                   v
                    |            [그래프 임베딩]
                    |                   |
                    |                   |
                    +---------+---------+
                              |
                              v
                      [편향도 예측 모델]
                          (XGBoost)
                              |
                              v
                         [추천 시스템]
                      (검색 + MMR 리랭킹)
                              |
                              v
                       [평가 & 성능 검증]
```

---

## 🛠️ 기술 스택

### Core Framework
- **Python 3.8+**
- **XGBoost** (편향도 예측 모델)
- **Scikit-learn** (Evaluation)

### Data Processing
- **Pandas** (데이터 처리)
- **NumPy** (수치 연산)
- **BeautifulSoup4** (웹 크롤링)

### Visualization & Analysis
- **Matplotlib** (시각화)
- **tqdm** (진행바)

### Workflow Management
- **Apache Airflow** (데이터 파이프라인)
- **BigQuery** (데이터 웨어하우스)

---

## 📁 디렉토리 구조

- `data/` 및 `dags/` 디렉토리는 윤리적 고려사항으로 인해 공개하지 않습니다.

```
politics_article_recommendation/
│
├── src/                            # 소스 코드
│   ├── bias_scoring/               # 편향도 예측
│   │   ├── apply_labels.py         # 언론사 레이블 적용
│   │   ├── text_encoder.py         # 텍스트 임베딩
│   │   ├── train_baseline_model.py # Baseline 모델 (텍스트만)
│   │   ├── train_full_model.py     # Full 모델 (텍스트+그래프)
│   │   ├── predict_all.py          # 전체 기사 예측
│   │   └── media_bias_map.py       # 언론사 편향도 매핑
│   │
│   ├── graph/                      # 그래프 임베딩
│   │   ├── graph_features.py       # Edge list 생성 & Feature 추출
│   │   └── embedding_loader.py     # Node2Vec 그래프 임베딩
│   │
│   ├── recommendation/             # 추천 시스템
│   │   ├── candidate_retrieval.py  # 후보 검색 (Balanced-coverage)
│   │   ├── mmr_reranking.py        # MMR 리랭킹 (Coverage-MMR)
│   │   ├── diversity_metrics.py    # 다양성 지표 (ILD, Coverage, RR-ILD)
│   │   ├── final_recommender.py    # 최종 통합 추천 시스템
│   │   └── test_pipeline.py        # 통합 테스트
│   │
│   └── evaluation/                 # 평가
│       ├── ablation_study.py       # Ablation Study
│       ├── bias_evaluation.py      # 편향도 예측 평가
│       └── recommendation_eval.py  # 대규모 추천 평가 (1000 queries)
│
├── scripts/                        # 유틸리티 스크립트
│   └── search_and_recommend.py     # 기사 검색 및 추천 테스트
│
├── utils/                          # 공통 유틸리티
│   └── data_loader.py              # 데이터 로딩
│
├── news_crawler.py                 # 네이버 뉴스 크롤러
├── preprocessor.py                 # 데이터 전처리
├── requirements.txt                # 패키지 의존성
├── roadmap.md                      # 개발 로드맵
└── README.md                       # 프로젝트 문서
```
---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/sngsuNoh/politics_article_recommendation.git
cd politics_article_recommendation

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

본 프로젝트는 데이터를 포함하지 않습니다. 직접 데이터를 수집하려면:

```bash
# 1. 뉴스 크롤링
python news_crawler.py

# 2. 전처리
python preprocessor.py

# 3. 언론사 레이블 적용
python src/bias_scoring/apply_labels.py
```

### 3. 모델 학습

```bash
# 텍스트 임베딩 생성
python src/bias_scoring/text_encoder.py

# 그래프 구조 생성
python src/graph/graph_features.py

# 그래프 임베딩 생성
python src/graph/embedding_loader.py

# 편향도 예측 모델 학습
python src/bias_scoring/train_full_model.py

# 전체 기사 편향도 예측
python src/bias_scoring/predict_all.py
```

### 4. 추천 시스템 실행

```bash
# 추천 시스템 통합 테스트
python src/recommendation/test_pipeline.py

# 대규모 평가 (1000 queries)
python src/evaluation/recommendation_eval.py

# 최종 추천 시스템
python src/recommendation/final_recommender.py

# 인터랙티브 검색 및 추천
python scripts/search_and_recommend.py
```

---

## 🔍 추천 알고리즘

### Phase 1: 후보 검색 (Balanced-Coverage Retrieval)

**목표**: 진보/중립/보수 진영별로 균형 잡힌 후보 30개 선택

**알고리즘**:
```python
# 진영별 쿼터 설정
min_prog_ratio = 0.2  # 진보 20%
min_neut_ratio = 0.2  # 중립 20%
min_cons_ratio = 0.2  # 보수 20%

# 텍스트 + 그래프 임베딩 가중 결합
similarity = 0.6 × text_similarity + 0.4 × graph_similarity

# 각 진영에서 유사도 높은 순으로 쿼터 채움
# 나머지 40%는 전체 유사도 순으로 채움
```

**특징**:
- 쿼리 편향과 무관하게 다양한 관점 후보 확보
- 동일 언론사 기사 최대 3개로 제한
- 중복 기사 제거 (제목 일치 + 유사도 0.99 이상)

### Phase 2: MMR 리랭킹 (Coverage-MMR)

**목표**: 관련성과 다양성을 동시에 고려한 최종 5개 선택

**알고리즘**:
```python
# Phase 1: 진영별 Coverage 보장
진보 후보 중 관련성 최고 1개 선택
중립 후보 중 관련성 최고 1개 선택
보수 후보 중 관련성 최고 1개 선택

# Phase 2: 나머지 2개는 MMR로 선택
for 남은 슬롯:
    mmr_score = λ × relevance + (1-λ) × min_bias_diversity
    최고 mmr_score 기사 선택
```

**λ 파라미터**:
- **λ=0.5** (채택): 관련성과 다양성 균형
- λ=1.0: 관련성만 (순수 유사도 정렬)
- λ=0.0: 다양성만 (편향도 차이 최대화)

### 핵심 설계 결정

| 요소 | 선택 | 이유 |
|------|------|------|
| **검색 방식** | Balanced-coverage | 후보 단계에서 진영 균형 보장 |
| **리랭킹** | Coverage-MMR | 진보/중립/보수 각 1개 강제 포함 |
| **λ 값** | 0.5 | 평가에서 목표 초과 달성 |
| **후보 수** | 30개 | 다양성 확보하면서도 관련성 유지 |
| **최종 추천** | 5개 | 사용자 피로도 최소화 |

---

## 📊 평가 지표

### 1. 편향도 예측 성능

| 지표 | 값 | 의미 |
|------|-----|------|
| **MAE** | 7.70 | 평균 오차 7.70점 (±100 스케일) |
| **RMSE** | 15.55 | 제곱근 평균 제곱 오차 |
| **R²** | 0.270 | 분산 설명력 (레이블 특성상 제한적) |

**R² 해석**: 우리 데이터는 언론사별 고정 레이블을 사용하므로, 같은 언론사 내 기사들의 레이블이 모두 동일합니다. 이러한 "계단형" 레이블 구조에서는 R²가 구조적으로 낮게 나올 수밖에 없으며, MAE가 더 신뢰할 수 있는 지표입니다. XGBoost 모델은 텍스트 임베딩(768-dim)과 그래프 정보(임베딩 64-dim + Feature 5-dim)를 결합하여 총 837차원의 입력을 학습합니다.

### 2. 다양성 지표

**ILD (Intra-List Diversity)**
- 추천 리스트 내 모든 쌍의 편향도 차이 평균
- 높을수록 다양한 관점 포함
- **목표: ≥20**, **달성: 21.96** ✓

**Coverage**
- 진보/중립/보수 3개 진영 포함 비율
- 1.0 = 세 진영 모두 포함
- **목표: ≥0.8**, **달성: 1.00** ✓

**RR-ILD (Rank & Relevance-sensitive ILD)**
- 상위 순위 + 높은 관련성 쌍의 다양성을 더 중시
- 사용자 경험에 가까운 지표
- **목표: ≥0.2**, **달성: 0.22** ✓

### 3. Ablation Study 결과

| 모델 | 구성 | Test MAE | Test R² |
|------|------|-----|-----|
| **Model A** | Text only (Baseline) | 8.91 | 0.156 |
| **Model B** | Text + Graph Feature | 7.82 | 0.261 |
| **Model C** | Text + Graph Emb | 8.85 | 0.160 |
| **Model D** | Text + Graph Emb + Feature (Full) | **7.70** | **0.270** |

**결론**: 그래프 정보(임베딩 64-dim + Feature 5-dim) 추가로 MAE 13.6% 개선 (8.91 → 7.70)

---

## 🎯 실험 결과

### 대규모 평가 (1000 queries)

```
추천 시스템 성능:
  ILD:        21.96 ± 11.31  (목표: ≥20.0)   ✓ PASS
  Coverage:   1.00 ± 0.00    (목표: ≥0.8)    ✓ PASS
  RR-ILD:     0.22 ± 0.10    (목표: ≥0.2)    ✓ PASS

처리 속도:
  1000개 쿼리 처리: 67초
  쿼리당 평균: 67ms (실시간 서비스 가능)

Coverage 달성률:
  1000/1000 쿼리에서 Coverage = 1.0 달성 (100%)
```

### 주요 발견

**1. 주제별 ILD 변동**
- 정치 스캔들: ILD 40~60 (관점이 극명하게 갈림)
- 지역 행사: ILD 10~15 (대부분 언론사가 유사하게 보도)
- 경제 정책: ILD 25~35 (중간 수준)

**2. Coverage-MMR의 효과**
- 모든 쿼리에서 진보/중립/보수 3개 진영 포함 보장
- 쿼리 편향과 무관하게 안정적 성능 유지

**3. λ 변화의 영향**
- Coverage-MMR 사용 시 λ=0.3~0.7 범위에서 결과 유사
- Phase 1에서 3개 진영 강제 선택이 지배적 역할

---

## 📚 참고 문헌

- **Filter Bubble**: Pariser, E. (2011). The filter bubble: What the Internet is hiding from you.
- **kcbert**: Junbum Lee. (2020). KcBERT: Korean comments BERT.
- **MMR**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.