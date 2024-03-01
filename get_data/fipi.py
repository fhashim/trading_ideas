from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time as t
from selenium import webdriver
from datetime import datetime
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import sqlalchemy as sa


# dates = pd.bdate_range('2019-05-15', '2020-05-15')
# dates = pd.bdate_range('2021-04-27', '2021-05-03')
dates = pd.bdate_range('2021-07-19', '2021-08-05')
# dates = pd.bdate_range('2021-01-01', '2021-03-28')

import cloudscraper 
 
scraper = cloudscraper.create_scraper(delay=10, browser="chrome") 
scraper = cloudscraper.create_scraper(delay = 10,
    browser={
        'browser': 'firefox',
        'platform': 'windows',
        'mobile': False
    }
)
print(scraper.get("https://webcache.googleusercontent.com/search?q=cache:www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise"))


import cloudscraper
import requests
ses = requests.Session()

ses.headers = {
    'referer': 'https://magiceden.io/',
    'accept': 'application/json'
}

ses.headers = {
    'accept': 'application/json'
}

ses.headers = {
    'referer': 'https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
    'accept': 'application/json'
}
scraper = cloudscraper.create_scraper(sess=ses)
hookLink = f"https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise"
meG = scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)
print(meG.status_code)
print(meG.text)

content = scraper.get("https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise").text 
 
print(content)

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)
print(scraper.get("https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise"))

# pip install zenrows
from zenrows import ZenRowsClient

client = ZenRowsClient("192e0e32b079e1e017e702db31692191928835ba")
url = "https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise"

response = client.get(url)

print(response.text)

BeautifulSoup.prettify(response)

import cfscrape 
 
scraper = cfscrape.create_scraper() 
scraped_data = scraper.get("https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise") 
print(scraped_data.text)


t0 =t.time()

for d in dates:
    try: 
        date = d.strftime('%d-%m-%y')

        conv_date = datetime.strptime(date, '%d-%m-%y')

        month = conv_date.strftime("%b")
        year = conv_date.strftime("%Y")
        day_name = conv_date.strftime("%A")
        day = conv_date.strftime("%d")
        day = day.lstrip("0")
        month_full_name = conv_date.strftime("%B")
        string = 'Select '+ day_name + ', ' + month + ' ' + day + ', ' + year 
        # print(string)
        # print(month_full_name)
        # print(year)

        options = Options()
        # options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(options=options)
        driver.get('https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise') 
        # driver.get('https://www.nccpl.com.pk/en/portfolio-investments/fipi-sector-wise')
        picker = wait(response, 10).until(EC.presence_of_element_located((By.ID, 'popupDatepicker')))
        driver.execute_script('arguments[0].scrollIntoView();', picker)
        picker.click()
        select_year = Select(wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="Change the year"]'))))
        select_year.select_by_visible_text(year)
        select_month = Select(wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="Change the month"]'))))
        select_month.select_by_visible_text(month_full_name)
        wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="{}"]'.format(string)))).click()
        search_button = driver.find_element_by_class_name('search_btn')
        search_button.click()

        #addition
        picker = wait(driver, 10).until(EC.presence_of_element_located((By.ID, 'popupDatepicker1')))
        driver.execute_script('arguments[0].scrollIntoView();', picker)
        picker.click()
        select_year = Select(wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="Change the year"]'))))
        select_year.select_by_visible_text(year)
        select_month = Select(wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="Change the month"]'))))
        select_month.select_by_visible_text(month_full_name)
        wait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[title="{}"]'.format(string)))).click()
        search_button = driver.find_element_by_class_name('search_btn')
        search_button.click()

        htmlSource = driver.page_source
        soup = BeautifulSoup(htmlSource, 'html.parser')
        #ActionChains(driver).move_to_element(search_button).click().perform()
        data = pd.read_html(htmlSource)
        df_lipi = data[0]



        df_lipi.to_csv(r'E:\Prices\FIPI_SECTOR_WISE\{}.csv'.format(date), index=False)


        df_lipi = pd.read_csv(r'E:\Prices\FIPI_SECTOR_WISE\{}.csv'.format(date))



        todays_date = conv_date

        df_lipi.columns = ['CLIENT_TYPE', 'SEC_CODE', 'SECTOR_NAME', 'MARKET_TYPE',
                           'BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                           'NET_VOLUME', 'NET_VALUE', 'USD']



        df_lipi.loc[:,'Date'] = todays_date

        df_lipi = df_lipi[['CLIENT_TYPE', 'Date', 'SEC_CODE', 'SECTOR_NAME', 'MARKET_TYPE',
                           'BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                           'NET_VOLUME', 'NET_VALUE', 'USD']]

        lipi_columns = ['BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                        'NET_VOLUME', 'NET_VALUE', 'USD']

        df_lipi.drop(df_lipi[df_lipi.MARKET_TYPE == 'TOTAL'].index, inplace=True)

        df_lipi.iloc[-1:,0] = 'FIPI_NET'

        df_lipi.iloc[-1:,2:4] = ['x9999', 'All']

        for string in lipi_columns:
            df_lipi[[string]] = (df_lipi[[string]].replace( '[\$,)]','', regex=True )
                               .replace( '[(]','-',   regex=True ).astype(float))


        engine_lipi = sa.create_engine('mssql+pyodbc://SA:Yasin@0334@localhost/' +
                                       'PSX?driver=ODBC Driver 17 for SQL Server')

        connection_lipi = engine_lipi.connect()

        tsql_chunksize = 2097 // len(df_lipi.columns)

        tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize

        df_lipi.to_sql('FIPI_SECTOR_WISE',engine_lipi, if_exists='append',index=False) #chunksize=tsql_chunksize , 


        #df_lipi.dtypes
        #print(df_lipi.head())
        #df_lipi.to_csv('/home/furqan/Desktop/fipi_sector_test01.csv', index = False)

        #df_lipi.to_csv('/home/furqan/Desktop/FIPI_SECTOR_WISE/test.csv')
    except:
        print(d)



t1 =t.time()
print("Exec time is: ", t1-t0)
