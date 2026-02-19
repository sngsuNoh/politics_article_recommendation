import sys
import os
import pendulum
from airflow import DAG 
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.news_crawler import crawl_naver_politics_by_date
from src.preprocessor import preprocess_daily_news

kst = pendulum.timezone("Asia/Seoul")

# 기본 설정
default_args = {
    'owner': 'shinji',
    'depends_on_past': False,
    'start_date' : datetime(2025, 8, 1, tzinfo=kst), # 수집 시작 날짜
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5) # 재시도 텀
}

# DAG 정의
with DAG(
    'politics_news_daily', # DAG ID
    default_args=default_args,
    description='매일 자정(KST), 어제 날짜의 뉴스를 수집 및 전처리',
    schedule_interval='0 0 * * *',
    catchup=True, 
    max_active_runs=1,
    tags=['politics', 'naver', 'project']
) as dag:

    cleanup_task = BashOperator(
        task_id='cleanup_zombie_process',
        bash_command='taskkill.exe /F /IM chromedriver.exe /T || true && taskkill.exe /F /IM chrome.exe /T || true'
    )

    crawling_task = PythonOperator(
        task_id='daily_crawl_task',

        # 실행할 파이썬 함수
        python_callable=crawl_naver_politics_by_date,

        # 날짜 전달
        op_kwargs={'target_date': '{{ next_ds_nodash }}'},

        execution_timeout=timedelta(minutes=40),
    )

    preprocessing_task = PythonOperator(
        task_id='daily_preprocess_task',
        python_callable=preprocess_daily_news,
        op_kwargs={'target_date': '{{ next_ds_nodash }}'},
    )

    # 작업 순서
    cleanup_task >> crawling_task >> preprocessing_task
