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

# Proxy Configuration
#proxy = "http://67.43.227.227:11023"  # Replace with your proxy server and port

# Create Chrome options instance
options = webdriver.ChromeOptions()
#options.add_argument(f"--proxy-server={proxy}")  # Add the proxy server argument
options.add_argument("--disable-blink-features=AutomationControlled")  # To prevent detection as a bot
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-extensions")  # Disable extensions
options.add_argument("--start-maximized")  # Open browser in maximized mode

# options.add_argument("--headless")  # Run headless, remove this if you want to see the browser

# Setting user-agent
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
options.add_argument(f"user-agent={user_agent}")

# Initialize WebDriver with Proxy
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Changing the property of the navigator value for webdriver to undefined
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# Custom headers
custom_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,it;q=0.6,es;q=0.5",
    "Cookie": "GarminUserPrefs=fr-FR; notice_behavior=expressed,eu; notice_gdpr_prefs=0:; notice_preferences=0:; notice_poptime=1619726400000; notice_poptime=1662667200000; notice_preferences=2:; notice_gdpr_prefs=0,1,2:; cmapi_cookie_privacy=permit 1,2,3; notice_behavior=implied,eu; cmapi_gtm_bl=ga-ms-ua-ta-asp-bzi-sp-awct-cts-csm-img-flc-fls-mpm-mpr-m6d-tc-tdc; _cfuvid=PCxP8JXP32ANQR2aQnfNw2LOWrATcfdNahbcRfoNWH4-1724400533133-0.0.1.1-604800000; __cfruid=676084269766438711271df7ccf654c8017e39d2-1724400534; GARMIN-SSO=1; GARMIN-SSO-CUST-GUID=c11660a0-752e-4d61-95e8-60e16367ad45; utag_main_v_id=0191375659c40020651403a52f4405075003406d00bd0; GarminNoCache=true; GARMIN-SSO-GUID=3EF17351F91C4E6288AAEBBA0ED1F55EB19A4015; SESSIONID=YTI0ZTkwMjctMjI1Ny00YTRlLWIxMTYtOWE5ZWY4YThlODhm; __cflb=02DiuJLbVZHipNWxN8wwnxZhF2QbAv3GYh7o4UFwkiGPN; TAsessionID=aa5d3a59-203b-4ddf-a0d5-ec5d03261bba|EXISTING; utag_main__sn=7; utag_main_ses_id=1724485662815%3Bexp-session; utag_main__ss=0%3Bexp-session; JWT_FGP=f9733f0e-5811-4e37-96f7-05b6c4273e6e; CONSENTMGR=c1:0%7Cc2:0%7Cc3:0%7Cc4:0%7Cc5:0%7Cc6:0%7Cc7:0%7Cc8:0%7Cc9:0%7Cc10:0%7Cc11:0%7Cc12:0%7Cc13:0%7Cc14:0%7Cc15:0%7Cts:1724486751020%7Cconsent:true; utag_main__se=3%3Bexp-session; utag_main__st=1724488551022%3Bexp-session; utag_main__pn=3%3Bexp-session; SameSite=None; ADRUM_BTa=R:38|g:304d768a-b3e3-4154-8f36-0b7f8f19ae51|n:garmin_869629ee-d273-481d-b5a4-f4b0a8c4d5a3; ADRUM_BT1=R:38|i:2694794|e:68|t:1724487302143"
}

# Function to set custom headers using DevTools Protocol
def set_custom_headers(driver, headers):
    # Enable the network domain, required for the next command
    driver.execute_cdp_cmd('Network.enable', {})

    # Set the custom headers
    driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {"headers": headers})

# Set the custom headers
set_custom_headers(driver, custom_headers)

# Open the Garmin Connect login page
#driver.get("https://sso.garmin.com/portal/sso/fr-FR/sign-in?clientId=GarminConnect&service=https://connect.garmin.com/modern/")
driver.get("https://connect.garmin.com/")

time.sleep(3)  # Wait for the page to load

### COKIES
# Add cookies using the Cookie header
#cookies_header = "GarminUserPrefs=fr-FR; notice_behavior=expressed,eu; notice_gdpr_prefs=0:; notice_preferences=0:; notice_poptime=1619726400000; ..."
# Add cookies using the Cookie header
# for cookie in cookies_header.split('; '):
#     if '=' in cookie:  # Check if the cookie has an '=' sign
#         name, value = cookie.split('=', 1)
#         driver.add_cookie({'name': name, 'value': value, 'domain': '.garmin.com'})
#     else:
#         print(f"Skipping invalid cookie: {cookie}")

# #Refresh the page to ensure cookies are applied
# driver.refresh()
### END COOKIES

# Maximize browser window
#driver.maximize_window()

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
