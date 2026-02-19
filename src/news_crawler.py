import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def crawl_naver_politics_by_date(target_date):
    """
    Airflow로부터 날짜(YYYYMMDD)를 받아서
    해당 날짜의 네이버 정치 뉴스를 상세하게 크롤링하여 저장하는 함수
    """
    print(f"[{target_date}] 상세 뉴스 수집을 시작합니다...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 네이버 정치(sid1=100) 뉴스 리스트 URL
    base_url = f'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=100&date={target_date}'
    
    news_list = []
    page = 1
    last_first_title = ""

    while True:
        url = f'{base_url}&page={page}'
        print(f"    - Page {page} 처리 중...", end="\r")
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 뉴스 리스트 가져오기
            news_items = soup.select('ul.type06_headline li, ul.type06 li')
            
            if not news_items:
                print(f"\n[{target_date}] 페이지 끝 도달 (항목 없음).")
                break
            
            # --- [페이지 중복 체크 로직] ---
            first_item = news_items[0]
            first_link_tag = first_item.select_one('dt:not(.photo) a, dd a')
            
            if first_link_tag:
                current_first_title = first_link_tag.text.strip()
                if current_first_title == last_first_title:
                    print(f"\n[{target_date}] 마지막 페이지 도달. (총 {page-1} 페이지)")
                    break
                last_first_title = current_first_title
            
            # --- [기사 상세 수집] ---
            for item in news_items:
                try:
                    link_tag = item.select_one('dt:not(.photo) a, dd a')
                    if not link_tag: continue
                    
                    article_url = link_tag.get('href')
                    if not article_url or 'naver.com' not in article_url: continue

                    # 1. 뉴스 ID 추출 (oid-aid)
                    news_id = ""
                    
                    # 패턴 A: oid=001&aid=0000001 형식
                    oid_match = re.search(r'oid=(\d+)', article_url)
                    aid_match = re.search(r'aid=(\d+)', article_url)
                    
                    if oid_match and aid_match:
                        news_id = f"{oid_match.group(1)}-{aid_match.group(1)}"
                    else:
                        # 패턴 B: /article/001/0000001 형식 (여기를 추가!)
                        path_match = re.search(r'/article/(\d+)/(\d+)', article_url)
                        if path_match:
                            news_id = f"{path_match.group(1)}-{path_match.group(2)}"
                    
                    title = link_tag.text.strip()
                    press = item.select_one('span.writing').text.strip() if item.select_one('span.writing') else ''
                    
                    # 2. 상세 페이지 진입 (본문, 기자명 등 추출)
                    article_resp = requests.get(article_url, headers=headers)
                    article_soup = BeautifulSoup(article_resp.text, 'html.parser')
                    
                    # 본문 추출 및 정제
                    content_tag = article_soup.select_one('#articleBodyContents, #dic_area, .article_body')
                    if content_tag:
                        # 불필요한 태그 제거
                        for tag in content_tag.select('script, div, span.end_photo_org, .img_desc'):
                            tag.decompose()
                        content = content_tag.get_text().strip()
                        content = re.sub(r'\n+', '\n', content)
                        content = re.sub(r'\s+', ' ', content)
                    else:
                        content = ''
                    
                    # 날짜 (작성/수정)
                    input_date = target_date # 기본값
                    modify_date = ''
                    
                    date_tag = article_soup.select_one('.media_end_head_info_datestamp_time, .article_info span')
                    if date_tag:
                        input_date = date_tag.get('data-date-time', date_tag.text.strip())
                        
                    modify_tag = article_soup.select_one('.media_end_head_info_datestamp_time.modify')
                    if modify_tag:
                        modify_date = modify_tag.get('data-modify-date-time', '')

                    # 기자명
                    reporter_tag = article_soup.select_one('.media_end_head_journalist_name, .journalist_name, .byline_s')
                    reporter = reporter_tag.text.strip() if reporter_tag else ''

                    # 이미지 URL
                    og_image = article_soup.select_one('meta[property="og:image"]')
                    image_url = og_image['content'] if og_image else ''
                    
                    # 원문 링크
                    origin_link_tag = article_soup.select_one('a.media_end_head_origin_link')
                    origin_url = origin_link_tag['href'] if origin_link_tag else ''

                    # 리스트에 추가
                    news_list.append({
                        'news_id': news_id,
                        '작성일시': input_date,
                        '수정일시': modify_date,
                        '언론사': press,
                        '기자명': reporter,
                        '제목': title,
                        '본문': content,
                        '이미지URL': image_url,
                        '원문링크': origin_url,
                        '네이버링크': article_url
                    })
                    
                except Exception:
                    continue # 개별 기사 에러는 무시하고 다음 기사로
            
            page += 1
            
        except Exception as e:
            print(f"\n페이지 처리 중 에러 발생: {e}")
            break
    
    # --- [데이터 저장] ---
    save_path = 'data/raw'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    filename = f'data/raw/naver_politics_{target_date}.csv'

    if news_list:
        df = pd.DataFrame(news_list)

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n저장 완료: {filename} (총 {len(news_list)} 건)")
    else:
        print(f"\n{target_date}일자 기사를 찾을 수 없습니다.")