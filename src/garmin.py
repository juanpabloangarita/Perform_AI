from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from random import uniform
from params import *

from selenium.common.exceptions import TimeoutException

from selenium_stealth import stealth
from webdriver_manager.core.os_manager import ChromeType


# Proxy Configuration
#proxy = "http://67.43.227.227:11023"  # Replace with your proxy server and port

# Create Chrome options instance
options = webdriver.ChromeOptions()
#options.add_argument(f"--proxy-server={proxy}")  # Add the proxy server argument
options.add_argument("--disable-blink-features=AutomationControlled")  # To prevent detection as a bot
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("--disable-gpu") #??
options.add_argument("--no-sandbox") #??
options.add_argument("--disable-extensions") #??  # Disable extensions
options.add_argument("--start-maximized")  # Open browser in maximized mode

# options.add_argument("--headless")  # Run headless, remove this if you want to see the browser

# Setting user-agent
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
options.add_argument(f"user-agent={user_agent}")

# Initialize WebDriver with Proxy
# Specify the exact version of ChromeDriver to match your Chrome browser
# This is the correct way to specify the version using install() method
# Specify the correct version of ChromeDriver
#chrome_driver_path = ChromeDriverManager(version="119.0.6045.105").install()
#chrome_driver_path = ChromeDriverManager().install()
#chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE, version="119.0.6045.105").install()
chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
ChromeDriverManager()

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
# Apply stealth settings
stealth(driver,
       user_agent=user_agent,
       languages=["en-US", "en"],
       vendor="Google Inc.",
       platform="MacIntel",
       webgl_vendor="Intel Inc.",
       renderer="Intel Iris OpenGL Engine",
       fix_hairline=True,
)

# Changing the property of the navigator value for webdriver to undefined
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# Custom headers
custom_headers = {
    "User-Agent": user_agent,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.garmin.com",
}
# Set additional common headers to mimic a typical browser session
common_headers = {
    "Origin": "https://connect.garmin.com",
    "DNT": "1",  # Do Not Track header, set as 1 to mimic typical privacy setting
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



# Open the Garmin Connect login page
driver.get("https://sso.garmin.com/portal/sso/en-GB/sign-in?clientId=GarminConnect&service=https%3A%2F%2Fconnect.garmin.com%2Fmodern")

time.sleep(3)  # Wait for the page to load

# Wait for the page to load and set cookies
time.sleep(3)

# # Extract and set cookies from a prior session or response
# cookies_header = "SESSION=NDQ5ZjE0MWEtMTU4ZS00YjUyLTlmNWYtNzM5ODgxMGFkMGEz; __VCAP_ID__=085b90dd-b842-40c2-7c5d-b12d; GarminUserPrefs=en-GB; __cflb=02DiuEofTPbaQ2tyBEVR6YcR5RQYPgpFZ1erLMDvdbay2; _cfuvid=ggLto8vv5yULggbEIcGQCJ5NI8mg8CYIzClkFJPsFKQ-1724605422525-0.0.1.1-604800000; cf_clearance=yhQqEnVG34w15p_3yO_YezG83zwF8vfJNnxGDMwwS9Q-1724672483-1.2.1.1-oYySv2jl4jC_LHsSDusSQKTAfEncKv_zwEjlW.Iq4mlHOJV1ED0BScCdO5qON4LvVgmZ3zpVCm8At6T4e7yXurhWqNJzx8ITK_vBvS3Mnpov._GFgki30NzRnlLuuYQLkFDFnkEjjwh6wUT_GkTvmr574M7V2copwUPDRCajuBIExfOy6y9oDq8FmGD20Jw5KjMDNTXfYmY5tXuClzOU1wGkxzrj_SnDUzlzfUAQFFqcGY9BrXgvy4PpQ6WZl_BmpN.chfFhDx6pspkeqZhNBuuJcmoK4KiZJEaLE.ZJlBo6ledBQVgbxs1hlsMGbQko1hbuWXuIIxGUzaRy6l61vxpYHopI6a7zBf1bnxiLBAa4V_2cinc4sBwObd0epBECf8YeRdgnQTO5S2iVxTrrIw"
# for cookie in cookies_header.split('; '):
#     if '=' in cookie:
#         name, value = cookie.split('=', 1)
#         driver.add_cookie({'name': name, 'value': value, 'domain': '.garmin.com'})
#     else:
#         print(f"Skipping invalid cookie: {cookie}")

# # Refresh the page to apply cookies and custom headers
# driver.refresh()

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

# Keep the browser window open
print("Script has completed. Press Enter to close the browser.")
input()  # Keeps the script running and browser open until Enter is pressed

# If you want to close manually after inspection
driver.quit()
