from selenium import webdriver
from selenium.common import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("start-maximized")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36")

# ChromeDriver 설정 (경로 수정 및 Raw String 사용)
# service = Service(executable_path=r"D:\chromedriver-win64\chromedriver.exe")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

# 자동화 감지 회피 스크립트 추가
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['ko-KR', 'ko'] });
    """
})

# 웹사이트 접속 (실제 URL로 변경)
driver.get("https://map.naver.com/v5/search")  # 예시로 네이버 지도를 사용합니다.

# 검색창이 로드될 때까지 대기
wait = WebDriverWait(driver, 10)
search_box = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "input_search")))

# 검색어 입력
search_box.send_keys("식당네오")
search_box.send_keys(Keys.RETURN)

# 검색 결과 로딩 대기 (필요한 경우 시간 조정)
time.sleep(random.uniform(2, 4))

# 왼쪽 iframe으로 전환
def switch_left():
    driver.switch_to.default_content()  # 최상위 컨텍스트로 이동
    iframe = driver.find_element(By.ID, "searchIframe")
    driver.switch_to.frame(iframe)

# 오른쪽 iframe으로 전환
def switch_right():
    driver.switch_to.default_content()
    iframe = driver.find_element(By.ID, "entryIframe")
    driver.switch_to.frame(iframe)

# 왼쪽 iframe으로 전환하여 요소 찾기
switch_left()
# next_page = driver.find_element(By.XPATH, '//*[@id="app-root"]/div/div[3]/div[2]/a[7]').get_attribute(
#         'aria-disabled')
try:
    # 요소가 나타날 때까지 대기
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((
            By.XPATH, '//span[(contains(@class, "YwYLL") or contains(@class, "TYaxT")) and text()="식당네오"]'
        ))
    )
    # 스크롤하여 요소가 보이게 하기 (필요한 경우)

    driver.execute_script("arguments[0].scrollIntoView();", element)
    # 요소 클릭
    element.click()
    time.sleep(random.uniform(2, 4))

    switch_right()
    name_element = driver.find_element(By.XPATH, '//span[@class="GHAhO"]')
    print(name_element)
    name = name_element.text.strip()
    print(name)
    element_3 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="NSTUp"]'))
    )
    # 스크롤하여 요소가 보이게 하기 (필요한 경우)
    driver.execute_script("arguments[0].scrollIntoView();", element_3)
    reviews = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//a[@data-pui-click-code="rvshowmore"]')))

    for review in reviews:
        review_text = review.text
        print(review_text)

except ElementClickInterceptedException:
    switch_right()
    name_element = driver.find_element(By.XPATH, '//span[@class="GHAhO"]')
    print(name_element)
    name = name_element.text.strip()
    print(name)
    element_2 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="NSTUp"]'))
    )
    # 스크롤하여 요소가 보이게 하기 (필요한 경우)
    driver.execute_script("arguments[0].scrollIntoView();", element_2)

    reviews = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//a[@data-pui-click-code="rvshowmore"]')))

    for review in reviews:
        review_text = review.text
        print(review_text)




# 필요한 작업 수행 후 브라우저 닫기 (원하는 경우)
# driver.quit()