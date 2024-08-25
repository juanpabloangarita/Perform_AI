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
# options.add_argument("--headless")  # Run headless, remove this if you want to see the browser
print("ONE")
# Setting user-agent
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
options.add_argument(f"user-agent={user_agent}")
print("TWO")
# Initialize WebDriver with Proxy
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
print("THREE")
# Changing the property of the navigator value for webdriver to undefined
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
print("FOUR")
# Open the Garmin Connect login page
driver.get("https://sso.garmin.com/portal/sso/fr-FR/sign-in?clientId=GarminConnect&service=https://connect.garmin.com/modern/")
time.sleep(3)  # Wait for the page to load
print("FIVE")
# Add cookies using the Cookie header
cookies_header = "GarminUserPrefs=fr-FR; notice_behavior=expressed,eu; notice_gdpr_prefs=0:; notice_preferences=0:; notice_poptime=1619726400000; ..."
# Add cookies using the Cookie header
# for cookie in cookies_header.split('; '):
#     if '=' in cookie:  # Check if the cookie has an '=' sign
#         name, value = cookie.split('=', 1)
#         driver.add_cookie({'name': name, 'value': value, 'domain': '.garmin.com'})
#     else:
#         print(f"Skipping invalid cookie: {cookie}")

# #Refresh the page to ensure cookies are applied
# driver.refresh()
time.sleep(3)  # Wait for the page to reload
print("SIX")
# Maximize browser window
driver.maximize_window()
print("SEVEN")
# Interact with the login form
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
# # Wait for the login button to be clickable and then click it
# wait = WebDriverWait(driver, 10)
# print("TENrwo")


# #button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
# button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[not(@disabled) and @type='submit']")))

# # Additional sleep to give time for potential JavaScript validations
# time.sleep(2)

# # Scroll to make sure the button is visible
# driver.execute_script('arguments[0].scrollIntoView(true);', button)

# # Click the button
# button.click()
# print("TENthree")
# #button.click()
# print("TENfour")

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





















######################

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# from random import uniform
# from params import *

# # Create Chrome options instance
# options = webdriver.ChromeOptions()
# options.add_argument("--disable-blink-features=AutomationControlled")  # To prevent detection as a bot
# #options.add_argument("--headless")  # Run headless, remove this if you want to see the browser
# options.add_argument("--disable-gpu")
# options.add_argument("--no-sandbox")

# # Setting user-agent
# user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
# options.add_argument(f"user-agent={user_agent}")

# # Use the Cookie header for authentication
# cookies_header = (
#     "GarminUserPrefs=fr-FR; notice_behavior=expressed,eu; notice_gdpr_prefs=0:; notice_preferences=0:; "
#     "notice_poptime=1619726400000; notice_poptime=1662667200000; notice_preferences=2:; "
#     "notice_gdpr_prefs=0,1,2:; cmapi_cookie_privacy=permit 1,2,3; notice_behavior=implied,eu; "
#     "cmapi_gtm_bl=ga-ms-ua-ta-asp-bzi-sp-awct-cts-csm-img-flc-fls-mpm-mpr-m6d-tc-tdc; "
#     "_cfuvid=PCxP8JXP32ANQR2aQnfNw2LOWrATcfdNahbcRfoNWH4-1724400533133-0.0.1.1-604800000; "
#     "__cfruid=676084269766438711271df7ccf654c8017e39d2-1724400534; GARMIN-SSO=1; "
#     "GARMIN-SSO-CUST-GUID=c11660a0-752e-4d61-95e8-60e16367ad45; utag_main_v_id=0191375659c40020651403a52f4405075003406d00bd0; "
#     "GarminNoCache=true; GARMIN-SSO-GUID=3EF17351F91C4E6288AAEBBA0ED1F55EB19A4015; "
#     "SESSIONID=YTI0ZTkwMjctMjI1Ny00YTRlLWIxMTYtOWE5ZWY4YThlODhm; __cflb=02DiuJLbVZHipNWxN8wwnxZhF2QbAv3GYh7o4UFwkiGPN; "
#     "TAsessionID=aa5d3a59-203b-4ddf-a0d5-ec5d03261bba|EXISTING; utag_main__sn=7; utag_main_ses_id=1724485662815%3Bexp-session; "
#     "utag_main__ss=0%3Bexp-session; JWT_FGP=f9733f0e-5811-4e37-96f7-05b6c4273e6e; CONSENTMGR=c1:0%7Cc2:0%7Cc3:0%7Cc4:0%7Cc5:0%7Cc6:0%7Cc7:0%7Cc8:0%7Cc9:0%7Cc10:0%7Cc11:0%7Cc12:0%7Cc13:0%7Cc14:0%7Cc15:0%7Cts:1724486751020%7Cconsent:true; "
#     "utag_main__se=3%3Bexp-session; utag_main__st=1724488551022%3Bexp-session; utag_main__pn=3%3Bexp-session; "
#     "SameSite=None; ADRUM_BTa=R:38|g:304d768a-b3e3-4154-8f36-0b7f8f19ae51|n:garmin_869629ee-d273-481d-b5a4-f4b0a8c4d5a3; "
#     "ADRUM_BT1=R:38|i:2694794|e:68|t:1724487302143"
# )

# # Initialize WebDriver
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # Open the Garmin Connect login page
# driver.get("https://sso.garmin.com/portal/sso/fr-FR/sign-in?clientId=GarminConnect&service=https://connect.garmin.com/modern/")
# time.sleep(3)  # Wait for the page to load

# # Add cookies using the Cookie header
# for cookie in cookies_header.split('; '):
#     name, value = cookie.split('=', 1)
#     driver.add_cookie({'name': name, 'value': value, 'domain': '.garmin.com'})

# # Refresh the page to ensure cookies are applied
# driver.refresh()
# time.sleep(3)  # Wait for the page to reload

# # Check if login is required by looking for the email input field

# user = driver.find_element(By.ID, 'email')

# # Maximize browser window
# driver.maximize_window()

# # Find the email input field and enter the email
# user.send_keys(EMAIL)
# time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior

# # Scroll the window slightly
# driver.execute_script('window.scrollTo(0, 700)')
# time.sleep(uniform(1, 3))

# # Find the password input field and enter the password
# password = driver.find_element(By.ID, 'password')
# password.send_keys(PASSWORD)
# time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior

# # Scroll again
# driver.execute_script('window.scrollTo(30, 500)')
# time.sleep(uniform(1, 3))

# # Wait for the login button to be clickable and then click it
# wait = WebDriverWait(driver, 10)
# button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and not(@disabled)]")))
# button.click()

# # Wait for successful navigation or confirmation
# #success_message = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'desired-success-element')))
# #print("Logged in successfully:", success_message.text)

# # except Exception as e:
# #     print("Login not required or element not found, error:", e)

# # Navigate to a page requiring login to test if cookies work
# # driver.get("https://connect.garmin.com/modern/")
# # time.sleep(3)  # Wait for the page to load

# # Try to find the login button again, if applicable
# # try:
# #     button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' and not(@disabled)]")))
# #     print("Button found and is clickable:", button.text)
# #     button.click()
# # except Exception as e:
# #     print("Error finding or clicking the button:", e)

# # Close the browser when done
# #driver.quit()




























# # ####################################
# # # pylint: disable=missing-docstring,invalid-name

# # # $CHALLENGIFY_BEGIN
# # from selenium import webdriver
# # from selenium.webdriver.common.by import By
# # from selenium.webdriver.support.ui import WebDriverWait
# # from selenium.webdriver.support import expected_conditions as ec

# # from selenium import webdriver
# # from selenium.webdriver.common.by import By
# # from selenium.webdriver.support.ui import WebDriverWait
# # from selenium.webdriver.support import expected_conditions as EC

# # from bs4 import BeautifulSoup
# # import csv
# # from params import *
# # import time

# # # Create Chromeoptions instance
# # options = webdriver.ChromeOptions()
# # #options.add_argument("--headless")
# # options.add_experimental_option("detach", True)

# # # Code between triple hash comes from POINT 2 OF: https://www.zenrows.com/blog/selenium-avoid-bot-detection#disable-automation-indicator-webdriver-flags
# # ###
# # # Adding argument to disable the AutomationControlled flag
# # options.add_argument("--disable-blink-features=AutomationControlled")

# # # Exclude the collection of enable-automation switches
# # options.add_experimental_option("excludeSwitches", ["enable-automation"])

# # # Turn-off userAutomationExtension
# # options.add_experimental_option("useAutomationExtension", False)

# # ### DIFFERET EXTRACT
# # test_ua = 'Mozilla/5.0 (Windows NT 4.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36'
# # options.add_argument("--window-size=1920,1080")

# # options.add_argument(f'--user-agent={test_ua}')

# # options.add_argument('--no-sandbox')
# # options.add_argument("--disable-extensions")
# # ### DIFFERENT EXTRACT

# # # Setting the driver path and requesting a page
# # driver = webdriver.Chrome(options=options)

# # # Changing the property of the navigator value for webdriver to undefined
# # driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
# # ###

# # # driver.get("https://connect.garmin.com/") # too long cuz then need to click connect
# # # driver.get("https://sso.garmin.com/portal/sso/fr-FR/sign-in?clientId=GarminConnect&service=https%3A%2F%2Fconnect.garmin.com%2Fmodern") # weird colors
# # driver.get("https://sso.garmin.com/portal/sso/fr-FR/sign-in?clientId=GarminConnect&service=https://connect.garmin.com/modern/")
# # time.sleep(7)
# # wait = WebDriverWait(driver, 15)
# # driver.maximize_window()
# # # Wait 3.5 on the webpage before trying anything
# # #time.sleep(3.5)
# # user = driver.find_element(By.ID, 'email') # Open the inspector in Chrome and find the input id!
# # user.send_keys(EMAIL)
# # time.sleep(4.5)
# # driver.execute_script('window.scrollTo(0, 700)')
# # WebDriverWait(driver, 7)
# # password = driver.find_element(By.ID, 'password')
# # password.send_keys(PASSWORD)
# # time.sleep(4.5)
# # driver.execute_script('window.scrollTo(30, 500)')
# # WebDriverWait(driver, 9)
# # #button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='g__button' and @type='submit']")))
# # button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@disabled type='submit']")))
# # # Click the button
# # button.click()

# # # search_input.send_keys("chocolate")
# # # search_input.submit()

# # # wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@id='recipes']")))

# # # recipe_urls = []
# # # cards = driver.find_elements(By.XPATH, "//div[@class='recipe my-3']")
# # # print(f"Found {len(cards)} results on the page")
# # # for card in cards:
# # #     url = card.get_attribute('data-href')
# # #     url = url.replace("http://localhost", "https://recipes.lewagon.com")
# # #     recipe_urls.append(url)

# # # print(recipe_urls)

# # # recipes = []
# # # for url in recipe_urls:
# # #   print(f"Navigating to {url}")
# # #   driver.get(url)
# # #   wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@class='p-3 border bg-white rounded-lg recipe-container']")))

# # #   soup = BeautifulSoup(driver.page_source, 'html.parser')
# # #   name = soup.find('h2').string.strip()
# # #   cooktime = soup.find('span', class_='recipe-cooktime').text.strip()
# # #   difficulty = soup.find('span', class_='recipe-difficulty').text.strip()
# # #   price = soup.find('small', class_='recipe-price').attrs.get('data-price').strip()
# # #   description = soup.find('p', class_='recipe-description').text.strip()
# # #   recipes.append({
# # #     'name': name,
# # #     'cooktime': cooktime,
# # #     'difficulty': difficulty,
# # #     'price': price,
# # #     'description': description
# # #   })

# # # with open('data/recipes.csv', 'w') as file:
# # #   writer = csv.DictWriter(file, fieldnames=recipes[0].keys())
# # #   writer.writeheader()
# # #   writer.writerows(recipes)

# # # driver.quit()
# # # # $CHALLENGIFY_END
