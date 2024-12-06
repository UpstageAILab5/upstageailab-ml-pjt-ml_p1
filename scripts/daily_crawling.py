from selenium import webdriver      # 브라우저를 제어하는 데 사용
from selenium.common import NoSuchElementException, TimeoutException, ElementClickInterceptedException      # 발생할 수 있는 일반적인 예외들을 포함
from selenium.webdriver.common.by import By     # 요소를 찾는 방법(예: ID, 클래스 이름, XPath 등)을 정의
from selenium.webdriver.common.keys import Keys     # 키보드 입력을 시뮬레이션하는 데 사용되는 특수 키들을 정의
from selenium.webdriver.chrome.options import Options       # Chrome 브라우저의 옵션을 설정하는 데 사용
from selenium.webdriver.chrome.service import Service as ChromeService      # ChromeDriver 서비스를 관리하는 데 사용
from webdriver_manager.chrome import ChromeDriverManager        # ChromeDriver를 자동으로 다운로드하고 관리하는 데 사용
from selenium.webdriver.support.ui import WebDriverWait         # 웹 요소가 로드될 때까지 기다리는 등의 지원 기능을 제공
from selenium.webdriver.support import expected_conditions as EC        # 특정 조건이 만족될 때까지 기다리는 데 사용되는 예상 조건들을 정의
import time
import random
import csv
from datetime import datetime
import pandas as pd
import os

def daily_crawling(start_num, end_num):
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36")

    # CSV 파일 읽기
    df_date = pd.read_csv('data/store_recent_date.csv')
    df = pd.read_csv('data/store_list.csv')

    # 0부터 79까지의 식당 데이터를 순회
    for index in range(start_num, end_num):
        try:
            # 각 식당의 정보 가져오기
            name = str(df.iloc[index]['식당 이름'])
            address = str(df.iloc[index]['주소'])
            store_num = str(df.iloc[index]['식당 번호'])
            
            # 최근 리뷰 기준
            recent_name = df_date.loc[df_date['식당 번호'] == int(store_num), '작성자'].values[0]
            no_recent = False

            # 최근일자 datetime 객체인지 확인
            recent_date = df_date.loc[df_date['식당 번호'] == int(store_num), '최근 날짜'].values[0]
            recent_date = ' '.join(recent_date.split()[:3])
            recent_date = datetime.strptime(recent_date, '%Y년 %m월 %d일')                
            
            print(f"\n=== 크롤링 시작: {name} ({index+1}/80) ===\n")
            
            # ChromeDriver 설정
            driver = webdriver.Chrome(options=chrome_options)  

            # 자동화 감지 회피 스크립트
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['ko-KR', 'ko'] });
                """
            })
            
            # 웹사이트 접속
            # 데이터 정렬 최신순 조건 추가 
            driver.get(f"https://pcmap.place.naver.com/restaurant/{store_num}/review/visitor?reviewSort=recent")
            
            # 검색창이 로드될 때까지 대기
            wait = WebDriverWait(driver, 10)
            time.sleep(random.uniform(2, 4))   

            # 리뷰 요소 찾기 > 리뷰 작성일자 확인 > df_date 파일의 최근 작성일자와 비교 > 작성자 확인 > 추가 리뷰 크롤링  
            while True:
                temp_df = pd.DataFrame(columns=['상호명', '주소', '작성자', '평점들', '방문일', '리뷰내용', '태그들'])
                review_containers = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "pui__X35jYm")))                    
                
                for index_num, container in enumerate(review_containers):
                    try:
                        # 방문일
                        visit_dates = container.find_elements(By.CLASS_NAME, "pui__QKE5Pr")
                        dates_raw = [date.text for date in visit_dates]
                        
                        # 방문일 데이터 분리
                        visit_info = dates_raw[0].split('\n') if dates_raw else [''] * 5

                        # 방문일 '요일' 제거
                        update_date = ' '.join(visit_info[2].split()[:3])
                        update_date = datetime.strptime(update_date, '%Y년 %m월 %d일')

                        # 작성자
                        writer_name = container.find_element(By.CLASS_NAME, "pui__NMi-Dp").text

                        if update_date > recent_date:
                            print(f"기존 {recent_date} 보다 {update_date} 최근 리뷰가 있습니다.")
                        elif update_date == recent_date:
                            # 방문일이 최근 날짜와 같다면, 작성자 비교
                            if writer_name != recent_name:
                                print(f"기존 {recent_date} 보다 {writer_name} 최근 리뷰가 있습니다.")
                            else:
                                print(f"기존 {recent_date} 이후 추가 리뷰가 없습니다.")
                                no_recent = True
                                break
                        else:
                            print(f"기존 {recent_date} 이후 추가 리뷰가 없습니다.")
                            no_recent = True
                            break

                        # 개별 리뷰의 더보기 버튼 처리
                        try:
                            more_button = container.find_element(By.CSS_SELECTOR, "a.pui__jhpEyP.pui__ggzZJ8")
                            driver.execute_script("arguments[0].click();", more_button)
                            time.sleep(random.uniform(0.5, 1))
                        except NoSuchElementException:
                            pass

                        # 각 요소 추출                    
                        # 리뷰 개수
                        review_num = container.find_element(By.CLASS_NAME, "pui__WN-kAf").text                                                

                        # 리뷰내용
                        review_text = container.find_element(By.CLASS_NAME, "pui__xtsQN-").text
                        
                        # 태그들 (모두 펼쳐진 상태에서 가져오기)
                        tags = container.find_elements(By.CLASS_NAME, "pui__jhpEyP")
                        # 더보기 버튼 제외하고 태그만 가져오기
                        tags_text = [tag.text for tag in tags if 'ggzZJ8' not in tag.get_attribute('class')]
                        
                        print(f"작성자: {writer_name}")
                        print(f"리뷰개수: {review_num}")
                        print(f"방문일: {visit_info}")
                        print(f"리뷰: {review_text}")
                        print(f"태그: {tags_text}")
                        print("-" * 50)                    

                        temp_df.loc[index_num] = [name, address, writer_name, review_num, ' | '.join(visit_info), review_text, ' | '.join(tags_text)]

                        df_date.loc[df_date['식당 번호'] == int(store_num), '최근 날짜'] = visit_info[2]
                        df_date.loc[df_date['식당 번호'] == int(store_num), '작성자'] = writer_name
                        df_date.to_csv('data/store_recent_date.csv', index=False, encoding='utf-8')
                        recent_review = 1 
                                
                    except Exception as e:
                        print(f"개별 리뷰 처리 중 에러 발생: {e}")
                        continue

                try:
                    # no_recent = True 면 종료
                    if no_recent:
                        print("새로운 리뷰가 없습니다. 크롤링을 종료합니다.")
                        break
                        
                    more_reviews_button = driver.find_element(By.CLASS_NAME, "fvwqf")
                    driver.execute_script("arguments[0].click();", more_reviews_button)
                    print("더 많은 리뷰 불러오는 중...")
                    time.sleep(random.uniform(2, 3))
                except NoSuchElementException:
                    print("모든 리뷰를 불러왔습니다.")
                    break
                except ElementClickInterceptedException:
                    print("더보기 버튼 클릭 불가")
                    break

            # 현재 시간으로 파일명 생성 (식당별로 다른 파일)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = '/opt/airflow/scripts/review_data_daily'
            filename = f'reviews_{name}_{current_time}.csv'

            # 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(directory):
                os.makedirs(directory)

            temp_df.to_csv(f'{directory}/{filename}', index=False, encoding='utf-8')

            # 브라우저 종료
            driver.quit()
            
            # 식당 간 크롤링 간격
            time.sleep(random.uniform(5, 10))
            
        except Exception as e:
            print(f"식당 {name} 크롤링 중 에러 발생: {e}")
            if 'driver' in locals():
                driver.quit()
            continue

    print("\n=== 모든 식당 크롤링 완료 ===\n")