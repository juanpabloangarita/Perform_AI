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
import chromedriver_autoinstaller

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

    # NOTE: SETTING UP CHROME DRIVER ALTERNATIVE 1
    # chrome_driver_path = ChromeDriverManager().install()
    # service = Service(chrome_driver_path)
    # driver = webdriver.Chrome(service=service, options=options)

    # NOTE: SETTING UP CHROME DRIVER ALTERNATIVE 2
    # os.system('webdriver-manager install chrome')
    # chrome_driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()  # Specify Google Chrome
    # service = Service(chrome_driver_path)
    # driver = webdriver.Chrome(service=service, options=options)

    # NOTE: SETTING UP CHROME DRIVER ALTERNATIVE 3
    chromedriver_autoinstaller.install()
    driver = webdriver.Chrome(options=options)

    print()
    print()
    print(f"FINISHING SETUP_DRIVER")
    print()
    print()

    return driver


# Function to add cookies to the driver
def add_cookies(driver, cookies):
    for cookie in cookies:
        driver.add_cookie(cookie)

# Load cookies into the driver
def load_cookies():
    return [
        {"name": "OptanonAlertBoxClosed", "value": "2024-07-25T10:44:28.591Z", "domain": "home.trainingpeaks.com"},
        {"name": "_gcl_au", "value": "1.1.390963292.1725263977", "domain": "home.trainingpeaks.com"},
        {"name": "__RequestVerificationToken", "value": "<your_token_here>", "domain": "home.trainingpeaks.com"},  # Ensure to replace this with a valid token
        {"name": "OptanonConsent", "value": "<your_consent_value>", "domain": "home.trainingpeaks.com"}  # Update this as necessary
    ]


def accept_cookies_popping_up(driver):
    try:
        # Wait for the cookie consent button to appear and click it
        cookie_accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"))
        )
        cookie_accept_button.click()
        print("Accepted cookies.")
    except TimeoutException:
        print("No cookie consent popup detected.")


def wait_for_overlay_to_disappear(driver):
    try:
        # Wait for the overlay to become invisible (you can adjust the timeout)
        WebDriverWait(driver, 10).until(
            EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.onetrust-pc-dark-filter"))
        )
        print("Overlay disappeared.")
    except TimeoutException:
        print("The overlay did not disappear in time.")

# version 1
def click_calendar_button_1(driver):
    try:
        # Wait for and click the calendar button
        calendar_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "calendar"))
        )
        calendar_button.click()
    except TimeoutException:
        print("Calendar button not found or clickable")


# version 2
def click_calendar_button_2(driver):
    try:
        # Attempt to click the calendar button
        calendar_button = driver.find_element(By.CSS_SELECTOR, "button.appHeaderMainNavigationButtons.calendar")
        calendar_button.click()
        print("Clicked calendar button.")
    except ElementClickInterceptedException:
        # Fallback: Force the click using JavaScript
        print("Calendar button click intercepted, using JavaScript to force click.")
        driver.execute_script("arguments[0].click();", calendar_button)
    except NoSuchElementException:
        print("Calendar button not found.")



# date_option can be 'today', 'yesterday', or 'both'
def get_todays_activities(driver):
    return select_activities_by_date(driver, 'today')


def get_yesterdays_activities(driver):
    return select_activities_by_date(driver, 'yesterday')


def get_both_activities(driver):
    return select_activities_by_date(driver, 'both')


def get_tomorrow_activities(driver):
    return select_activities_by_date(driver, 'tomorrow')


# Function to get the activities for a specific date range
def select_activities_by_date(driver, date_option):
    accept_cookies_popping_up(driver)
    wait_for_overlay_to_disappear(driver)
    click_calendar_button_2(driver)

    # Get today's and yesterday's dates
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(1)).strftime('%Y-%m-%d')

    if date_option == 'today':
        return fetch_activities_for_date(driver, today)
    elif date_option == 'yesterday':
        return fetch_activities_for_date(driver, yesterday)
    elif date_option == 'tomorrow':
        return fetch_activities_for_date(driver, tomorrow)
    elif date_option == 'both':
        today_activities = fetch_activities_for_date(driver, today)
        yesterday_activities = fetch_activities_for_date(driver, yesterday)
        return today_activities + yesterday_activities
    else:
        raise ValueError("Invalid date option. Choose 'today', 'yesterday', or 'both'.")

def scroll_to_previous_week(driver):
    # Scroll up to go to the previous week
    # NOTE: Attempts to simulate scrolling by moving the mouse cursor, which may not trigger the actual scroll event.
    ActionChains(driver).move_by_offset(0, -300).perform()  # Scroll up
    time.sleep(1)  # Wait for the content to load


def scroll_up(driver):
    # NOTE: Directly scrolls the page using JavaScript, effectively updating the visible content.
    driver.execute_script("window.scrollBy(0, -300);")  # Scroll up
    time.sleep(1)  # Wait for content to load


def safe_find_element(driver, by, value, timeout=10, max_retries=3):
    """ Safely find an element with retry logic for stale references. """
    retries = 0
    while retries < max_retries:
        try:
            return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))
        except StaleElementReferenceException:
            print(f"StaleElementReferenceException encountered. Retrying... attempt {retries + 1}")
            retries += 1
            time.sleep(1)  # Small delay before retrying

    raise Exception(f"Element not found after {max_retries} retries due to stale element reference.")


def scroll_and_retry(driver, activity_locator, max_retries=3, scroll_amount=-300, delay=1):
    retries = 0
    while retries < max_retries:
        try:
            # Perform the scroll action using JavaScript
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(delay)  # Wait for content to load

            # Use the safe find element method to handle stale elements after scrolling
            activity = safe_find_element(driver, By.CSS_SELECTOR, activity_locator)

            # Perform any action on the re-located element
            return activity

        except StaleElementReferenceException as e:
            print(f"StaleElementReferenceException during scrolling: Retrying... attempt {retries + 1}")
            retries += 1
            time.sleep(1)  # Small delay before retrying

    raise Exception(f"Failed to locate element after {max_retries} attempts due to stale element reference.")


def safe_find_child_element(parent_element, by, value, max_retries=3, delay=1):
    """ Safely find a child element of a given parent element with retry logic for stale references. """
    retries = 0
    while retries < max_retries:
        try:
            return parent_element.find_element(by, value)
        except StaleElementReferenceException:
            print(f"StaleElementReferenceException: Retrying... attempt {retries + 1}")
            retries += 1
            time.sleep(delay)  # Delay before retrying

    raise Exception(f"Failed to find child element after {max_retries} retries due to stale element reference.")


def fetch_activities_for_date(driver, date_str):
    # Scroll to the previous week
    # scroll_to_previous_week(driver) # NOTE: -> selenium.common.exceptions.MoveTargetOutOfBoundsException: Message: move target out of bounds
    # Get today's date as a datetime object

    today = datetime.now()
    # Check if today is a Monday
    if today.weekday() == 0 and date_str == (datetime.now() - timedelta(1)).strftime('%Y-%m-%d'):
        # scroll_up(driver) # prior version
        date_element = scroll_and_retry(driver, f"div.dayContainer[data-date='{date_str}']", max_retries=3, scroll_amount=-300, delay=1)
    else:
        # Use the safe_find_element function to get today's dayContainer
        date_element = safe_find_element(driver, By.CSS_SELECTOR, f"div.dayContainer[data-date='{date_str}']")


    # Step 3: Scroll into view if necessary
    ActionChains(driver).move_to_element(date_element).perform()

    try:
        # Use the safe method to find the 'activities' element within 'date_element'
        activities_element = safe_find_child_element(date_element, By.XPATH, ".//div[contains(@class, 'activities')]")
    except TimeoutException:
        print("Timed out waiting for the activities element.")


    # Compliance status mapping
    compliance_mapping = {
        'workoutComplianceStatus planned future': 'Future',
        'workoutComplianceStatus planned notCompliant isSkipped past': 'Fail',
        'workoutComplianceStatus planned complete fullyCompliant past': 'Success',
        'workoutComplianceStatus planned complete partiallyCompliant past': 'Average',
        'workoutComplianceStatus planned complete notCompliant past': 'Mediocre'
    }

    # Step 5: Extract relevant information from the activities element
    activities = []
    for activity in activities_element.find_elements(By.CSS_SELECTOR, "div.MuiCard-root.activity"):
        activity_info = {}

        # Get the workout compliance status class
        try:
            compliance_status_class = activity.find_element(By.CSS_SELECTOR, "div.workoutComplianceStatus").get_attribute("class")
            # Map the class to a more user-friendly term
            compliance_status = compliance_mapping.get(compliance_status_class, "Unknown")  # Default to 'Unknown' if class not found
        except Exception as e:
            compliance_status = "Unknown"  # Default value if the element is not found
            print(f"Error fetching compliance status: {e}")

        # Sport Type (Bike, Run, Swim)
        try:
            sport_type = activity.find_element(By.CSS_SELECTOR, "div.printOnly.sportType").get_attribute("innerText")
        except Exception as e:
            sport_type = 'Unknown'  # Default value if the element is not found
            print(f"Error fetching sport type: {e}")

        if sport_type in ['Bike', 'Run', 'Swim']:
            try:
                title = activity.find_element(By.CSS_SELECTOR, "span.newActivityUItitle").text
            except NoSuchElementException as e:
                title = ''  # Default value if the element is not found
                print(f"Error fetching title: {e}")

            try:
                duration = activity.find_element(By.CSS_SELECTOR, "div.duration span.value").text
            except NoSuchElementException as e:
                duration = 0.0  # Default value if the element is not found
                print(f"Error fetching duration: {e}")

            try:
                tss = activity.find_element(By.CSS_SELECTOR, "div.tss span.value").text
            except NoSuchElementException as e:
                tss = 0.0  # Default value if the element is not found
                print(f"Error fetching TSS: {e}")

            try:
                description = activity.find_element(By.CSS_SELECTOR, "p.description").text
            except NoSuchElementException as e:
                description = ''  # Default value if the element is not found
                print(f"Error fetching description: {e}")

            try:
                coach_comments = activity.find_element(By.CSS_SELECTOR, "p.coachComments").text
            except NoSuchElementException as e:
                coach_comments = ''  # Default value if the element is not found
                print(f"Error fetching coach comments: {e}")

            activity_info['Date'] = date_str
            activity_info['compliance_status'] = compliance_status
            activity_info['WorkoutType'] = sport_type
            activity_info['Title'] = title
            activity_info['WorkoutDescription'] = description
            activity_info['CoachComments'] = coach_comments
            activity_info['duration'] = duration
            activity_info['tss'] = tss
            # Append this activity info to the list of activities
            activities.append(activity_info)

    return activities


def navigate_to_login(to_do):
    # Create Chrome options instance
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    # Initialize WebDriver
    driver = setup_driver(options)

    # Open the login page
    driver.get("https://home.trainingpeaks.com/login")

    # Wait for the page to load
    driver.implicitly_wait(5)
    """
    # Set cookies before navigating further
    cookies = load_cookies()
    add_cookies(driver, cookies)

    # Refresh the page to apply the cookies
    driver.refresh()
    """

    # Set cookies after loading the page to ensure the domain matches
    cookies = load_cookies()
    for cookie in cookies:
        # Only add cookies that match the current domain
        if cookie['domain'] in driver.current_url:
            driver.add_cookie(cookie)

    # After adding cookies, refresh or navigate again if necessary
    driver.refresh()

    # Wait for the page to load
    time.sleep(uniform(3, 7))
    driver.execute_script('window.scrollTo(0, 700)')

    # Interact with the login form
    username_field = driver.find_element(By.ID, 'Username')
    username_field.send_keys(USER_TP)
    print("Entered username.")
    time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior

    # Enter the password
    password_field = driver.find_element(By.ID, 'Password')
    password_field.send_keys(PASSWORD_TP)
    print("Entered password.")
    time.sleep(uniform(2, 5))  # Randomized delay to mimic human behavior

    # Wait for the login button to be clickable and then click it
    wait = WebDriverWait(driver, 10)

    try:
        # Find the login button
        login_button = wait.until(EC.element_to_be_clickable((By.ID, 'btnSubmit')))
        print("Login button is clickable.")

        # Scroll to make sure the button is visible
        driver.execute_script('arguments[0].scrollIntoView(true);', login_button)

        # Click the button using JavaScript to avoid detection
        driver.execute_script("arguments[0].click();", login_button)
        print("Clicked the login button.")
    except TimeoutException:
        print("TimeoutException: The login button was not clickable within the allotted time.")
        print("Page source for debugging:", driver.page_source)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Wait for additional actions if needed
    time.sleep(5)

    if to_do == 'both':
        data_scraped = get_both_activities(driver)
    elif to_do == 'today':
        data_scraped = get_todays_activities(driver)
    elif to_do == 'tomorrow':
        data_scraped = get_tomorrow_activities(driver)
    elif to_do == 'yesterday':
        data_scraped = get_yesterdays_activities(driver)
    else:
        print("TRAINING PEAKS:")
        print("NO DATE TO SCRAPE WAS GIVEN")


    time.sleep(uniform(2, 5))
    driver.quit()
    return data_scraped

if __name__ == '__main__':
    print(navigate_to_login('both'))
