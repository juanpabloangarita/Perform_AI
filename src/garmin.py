# garmin.py

# Import necessary modules
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from random import uniform
from params import *  # Assumed to contain EMAIL and PASSWORD variables

from selenium.common.exceptions import TimeoutException

from selenium_stealth import stealth  # Stealth to avoid bot detection
from webdriver_manager.core.os_manager import ChromeType

from selenium.webdriver.chrome.options import Options
import tempfile
import os

# Set environment variable for browser (for debugging purposes)
os.environ["BROWSER"] = "chrome"
os.environ['WDM_SKIP_VERSION_CHECK'] = 'true'


def setup_driver(options, is_simple=True):
    # Set up the ChromeDriver
    if is_simple:
        chrome_driver_path = ChromeDriverManager().install()
        service = Service(chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Initialize WebDriver with Proxy
        #chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE, version="119.0.6045.105").install()
        chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
        driver = webdriver.Chrome(service=Service(chrome_driver_path, log_path='chromedriver.log'), options=options)
    return driver

# Function to add cookies to the driver
def add_cookies(driver, cookies):
    for cookie in cookies:
        driver.add_cookie(cookie)

# Load cookies into the driver
def load_cookies():
    # Replace this with actual values you have extracted
    cookies = [
        {"name": "SESSION", "value": "NzQ2YmI0MWItY2QzMC00MmQ0LWJiYzUtNzJmMDkzOTliY2Zh", "domain": "sso.garmin.com"},
        {"name": "__VCAP_ID__", "value": "47d0f134-e4b5-4348-7807-d782", "domain": "sso.garmin.com"},
        {"name": "cf_clearance", "value": "St2C9FjcQSCYsMS8yxJ3ZKfdvuIfHnHJZpcQ.hHY8gs-1723210272-1.0.1.1-z89QpiyE9u.2KASTfIxFMJ9uJ58PCUZ8mifePZ2MUpsvK2xeGIa40jQwPO2RHkG2XfjvpqT_tzgHPt9AfY8v0A", "domain": "sso.garmin.com"},
        {"name": "SameSite", "value": "None", "domain": "sso.garmin.com"},
        {"name": "__cflb", "value": "02DiuEofTPbaQ2tyBEVR6YcR5RQYPgpFY26dLh9KigZtG", "domain": "sso.garmin.com"},
        {"name": "__cf_bm", "value": "GvAmHBsN6UKnmabksl3eonhOef23gO8ucIpQ50V3zdo-1724868769-1.0.1.1-VUTNws1QwSXik1DcKdwvh1qo9yJLt2x1oT314pAROQOs5I3GifTNqcBAJ1WSSEjGq15CGWE4wu3UeUagSS9OvA", "domain": "sso.garmin.com"},
        {"name": "_cfuvid", "value": "Rc6d2uCACOfnGUwigboqXJcE8QUOkbBYPrGbEdhF7to-1724868769034-0.0.1.1-604800000", "domain": "sso.garmin.com"},
        {"name": "ADRUM_BTa", "value": "R:0|g:3a47900d-1754-4253-b2af-a6923b852afd", "domain": "sso.garmin.com"},
        {"name": "ADRUM_BT1", "value": "R:0|i:2270382|e:120|t:1724868781116", "domain": "sso.garmin.com"}
    ]
    return cookies


def navigate_to_login_advanced():
    # Proxy Configuration
    # proxy = "http://67.43.227.227:11023"  # Replace with your proxy server and port

    # Create Chrome options instance
    options = webdriver.ChromeOptions()
    # options.add_argument(f"--proxy-server={proxy}")  # Add the proxy server argument
    options.add_argument("--disable-blink-features=AutomationControlled")  # To prevent detection as a bot
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-extensions")  # Disable all extensions
    options.add_argument("--disable-popup-blocking")  # Disable popup blocking
    options.add_argument("--start-maximized")
    options.add_argument("--no-first-run")  # Prevents Chrome's first-run setup
    options.add_argument("--no-default-browser-check")  # Disables the default browser check
    options.add_argument("--disable-infobars")  # Disables infobars like "Chrome is being controlled by automated test software"
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # options.add_argument("--incognito")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-software-rasterizer")  # Add this line
    options.add_argument("--remote-debugging-port=9222")  # Add this line
    # How to get rid of "Choose your search engine" dialog in Chrome v.127
    options.add_argument("--disable-search-engine-choice-screen") # https://stackoverflow.com/questions/78798750/how-to-get-rid-of-choose-your-search-engine-dialog-in-chrome-v-127-on-selenium
    # options.add_argument("--headless")  # Run headless, remove this if you want to see the browser
    # options.add_argument("--auto-open-devtools-for-tabs") #not sure for the moment

    # Create a temporary directory for the user data
    temp_user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={temp_user_data_dir}")

    # Set the correct path to the Chrome binary
    options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    # Set up a user profile directory (optional, prevents saving of any profile info)
    options.add_argument("--user-data-dir=/tmp/temporary-chrome-profile")  # Use a temporary profile

    # Setting user-agent to the latest version from inspection
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36" #
    options.add_argument(f"user-agent={user_agent}")

    # Initialize WebDriver with Proxy
    driver = setup_driver(options, is_simple=False)
    # Apply stealth settings
    stealth(driver,
        user_agent=user_agent,
        languages=["en-GB", "en"],
        vendor="Google Inc.",
        platform="MacIntel",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        run_on_insecure_origins=True
    )

    # Add this new script execution below: ### NEW
    driver.execute_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-GB', 'en']});
        window.screen = {width: 1920, height: 1080, availWidth: 1920, availHeight: 1080};
        window.outerWidth = window.innerWidth;
        window.outerHeight = window.innerHeight;
    """)

    # Changing the property of the navigator value for webdriver to undefined
    # driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})") #

    # Custom headers
    custom_headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        # "Accept-Encoding": "gzip, deflate, br",
        # "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,it;q=0.6,es;q=0.5",
        "Cache-Control": "max-age=0", #
        "Sec-Fetch-Dest": "document", #
        "Sec-Fetch-Mode": "navigate", #
        # "Sec-Fetch-Site": "none",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1", #
        "Upgrade-Insecure-Requests": "1",
        "sec-ch-ua": "\"Chromium\";v=\"128\", \"Not;A=Brand\";v=\"24\", \"Google Chrome\";v=\"128\"", #
        "sec-ch-ua-mobile": "?0", #
        "sec-ch-ua-platform": "\"macOS\"", #
        "Referer": "https://www.garmin.com"
    }
    # Set additional common headers to mimic a typical browser session
    common_headers = {
        "Origin": "https://connect.garmin.com",
        "DNT": "1",  # Do Not Track header, set as 1 to mimic typical privacy setting,
        "Connection": "keep-alive"
    }

    # Combine custom and common headers
    all_headers = {**custom_headers, **common_headers}

    # Function to set custom headers using DevTools Protocol
    def set_custom_headers(driver, headers):
        # Enable the network domain, required for the next command
        driver.execute_cdp_cmd('Network.enable', {})

        # Set the custom headers
        driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {"headers": headers})

    # Set the custom headers
    set_custom_headers(driver, all_headers)

    # Mimic network conditions for real-world scenario ### NEW
    driver.execute_cdp_cmd('Network.emulateNetworkConditions', {
        'offline': False,
        'latency': 100,  # 100ms latency
        'downloadThroughput': 500 * 1024,  # 500 kb/s
        'uploadThroughput': 500 * 1024,  # 500 kb/s
    })

    # Set cookies before navigating
    cookies = load_cookies()
    # Open the Garmin Connect login page
    driver.get("https://sso.garmin.com/portal/sso/en-GB/sign-in?clientId=GarminConnect&service=https%3A%2F%2Fconnect.garmin.com%2Fmodern")
    # add cookies
    #add_cookies(driver, cookies)
    #time.sleep(3)  # Wait for the page to load
    # Refresh to apply cookies
    #driver.refresh()
    # # Wait for the page to load and set cookies
    #time.sleep(3)
    # Keep the browser window open
    print(driver.title)
    input()
    driver.quit()


# # Wait and continue with additional actions
# time.sleep(5)

# # Interact with the login form
# user = driver.find_element(By.ID, 'email')
# user.send_keys(EMAIL)
# print("EIGHT")
# time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior
# driver.execute_script('window.scrollTo(0, 700)')
# time.sleep(uniform(1, 3))
# print("NINE")

# password = driver.find_element(By.ID, 'password')
# password.send_keys(PASSWORD)
# print("TEN")
# time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior

# driver.execute_script('window.scrollTo(30, 500)')
# time.sleep(uniform(1, 3))
# print("eleven")
# # # Wait for the login button to be clickable and then click it
# wait = WebDriverWait(driver, 10)
# print("TENrwo")



# ##||
# try:
#     #button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and not(@disabled)]")))
#     button = driver.find_element(By.XPATH, "//button[@type='submit' and @data-testid='g__button' and not(@disabled)]")

#     print("TENthree")
#     # Scroll to make sure the button is visible
#     driver.execute_script('arguments[0].scrollIntoView(true);', button)

#     # Click the button using JavaScript
#     driver.execute_script("arguments[0].click();", button)
#     print("TENfour")
#     #button.click()
#     print("TENTENTEN")
# except TimeoutException as e:
#     print("TimeoutException: The button was not clickable within the allotted time.")
#     print("Page source for debugging:", driver.page_source)
# except Exception as e:
#     print(f"An error occurred: {e}")

# ##||


def navigate_to_login_simple():
    # Create Chrome options instance
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # options.add_argument("--headless")  # Uncomment this line to run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-search-engine-choice-screen")

    # Create a temporary user data directory
    temp_user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={temp_user_data_dir}")

    # Set the path to Chrome binary explicitly
    options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    # Set up the ChromeDriver with the specified Chrome binary
    driver = setup_driver(options)

    # Open the Garmin Connect login page
    driver.get("https://sso.garmin.com/portal/sso/en-GB/sign-in?clientId=GarminConnect&service=https%3A%2F%2Fconnect.garmin.com%2Fmodern")

    print(driver.title)
    input()
    driver.quit()


# Uncomment one of the functions below to test
# navigate_to_login_simple()
navigate_to_login_advanced()
