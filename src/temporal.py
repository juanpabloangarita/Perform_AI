import os
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType  # Keep this import if needed

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime, timedelta, date

from random import uniform

from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException, ElementClickInterceptedException


from selenium.webdriver.common.action_chains import ActionChains

from params import *

headless_mode = (CLOUD_ON == 'yes')

# Set environment variable for browser (for debugging purposes)
os.environ["BROWSER"] = "chrome"
os.environ['WDM_SKIP_VERSION_CHECK'] = 'true'

def setup_driver(options):
    # Add headless mode options if needed
    if headless_mode:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")  # Disables GPU hardware acceleration
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920,1080")
    options.add_argument("--disable-web-security")  # Disables web security
    temp_user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={temp_user_data_dir}")


    # Add options to prevent bot detection
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Set up the ChromeDriver
    # chrome_driver_path = ChromeDriverManager().install()
    # service = Service(chrome_driver_path)

    chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()  # Specify Google Chrome
    service = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=service, options=options)
    return driver


# Create Chrome options instance
options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")

# Initialize WebDriver
driver = setup_driver(options)

# Open the login page
driver.get("https://home.trainingpeaks.com/login")