# variable names or variable data
import os

# GIVEN_DATE = '2024-09-08'
from datetime import datetime

# Set today's date in the format 'YYYY-MM-DD'
GIVEN_DATE = datetime.today().strftime('%Y-%m-%d')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME') # same name as env file
#data bucket juan csv
#DATA_BUCKET_JUAN
EMAIL=os.getenv('EMAIL')
PASSWORD=os.getenv('PASSWORD')
USER_DATA_FILE=os.getenv('USER_DATA_FILE')
CODE_PROMO=os.getenv('CODE_PROMO')
CLOUD_ON="yes"
USER_TP=os.getenv('USER_TP')
PASSWORD_TP=os.getenv('PASSWORD_TP')
INSTANCE_ID=os.getenv('INSTANCE_ID')
#SSH_KEY_STR=os.getenv('SSH_KEY_STR').replace("\\n", "\n")
INSTANCE_IP=os.getenv('INSTANCE_IP')
SSH_KEY_STR=os.getenv('SSH_KEY_STR')
BEST_MODEL=os.getenv('BEST_MODEL')
