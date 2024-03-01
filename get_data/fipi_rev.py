import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from get_data.cnxn import server_access
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
import time as t
import pandas as pd
from datetime import datetime

pd.options.mode.chained_assignment = None


# dates = pd.bdate_range('2021-08-04', '2021-09-30')
# dates = pd.bdate_range('2021-11-02', '2021-11-02')
dates = pd.bdate_range('2024-02-28', '2024-02-28')

for d in dates:

    try:

        date = d.strftime('%d-%m-%y')
        conv_date = datetime.strptime(date, '%d-%m-%y')
        day = conv_date.strftime("%d").lstrip('0')
        date = '{}/{}/{}'.format(day, conv_date.strftime("%m"), conv_date.strftime("%Y"))

        # url = 'https://www.nccpl.com.pk/en/market-information/fipi-lipi/fipi-sector-wise'
        url = 'https://www.nccpl.com.pk/en/portfolio-investments/fipi-sector-wise'

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        browser = uc.Chrome(options=options)
        browser.get(url)
        picker = wait(browser, 10).until(EC.presence_of_element_located((By.ID, 'popupDatepicker')))
        t.sleep(2)
        browser.execute_script('arguments[0].scrollIntoView();', picker)
        picker.click()
        t.sleep(3)
        browser.execute_script(f'arguments[0].value = "{date}";', picker)
        # print('clicked the day!')
        t.sleep(5)

        picker = wait(browser, 10).until(EC.presence_of_element_located((By.ID, 'popupDatepicker1')))
        t.sleep(2)
        browser.execute_script('arguments[0].scrollIntoView();', picker)
        picker.click()
        t.sleep(4)
        browser.execute_script(f'arguments[0].value = "{date}";', picker)
        # print('clicked the second day!')
        t.sleep(1)
        search_button = browser.find_element(By.XPATH, '//button[@class="search_btn"]/parent::div')
        t.sleep(2)
        search_button.click()
        # print('clicked search!')
        t.sleep(10)
        dfs = pd.read_html(browser.page_source)
        df_lipi = dfs[0]
        browser.quit()

        todays_date = conv_date

        df_lipi.columns = ['CLIENT_TYPE', 'SEC_CODE', 'SECTOR_NAME', 'MARKET_TYPE',
                           'BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                           'NET_VOLUME', 'NET_VALUE', 'USD']

        df_lipi.loc[:, 'Date'] = todays_date

        df_lipi = df_lipi[['CLIENT_TYPE', 'Date', 'SEC_CODE', 'SECTOR_NAME', 'MARKET_TYPE',
                           'BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                           'NET_VOLUME', 'NET_VALUE', 'USD']]

        lipi_columns = ['BUY_VOLUME', 'BUY_VALUE', 'SELL_VOLUME', 'SELL_VALUE',
                        'NET_VOLUME', 'NET_VALUE', 'USD']

        df_lipi.drop(df_lipi[df_lipi.MARKET_TYPE == 'TOTAL'].index, inplace=True)

        df_lipi.iloc[-1:, 0] = 'FIPI_NET'

        df_lipi.iloc[-1:, 2:4] = ['x9999', 'All']

        for string in lipi_columns:
            df_lipi[[string]] = (df_lipi[[string]].replace('[\$,)]', '', regex=True)
                                 .replace('[(]', '-', regex=True).astype(float))

        # engine = server_access()

        # df_lipi.to_sql('tbl_fipi_sector_wise', engine, if_exists='append', index=False)

        # t.sleep(10)

    except:

        print(d)
