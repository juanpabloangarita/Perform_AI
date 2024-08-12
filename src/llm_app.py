import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Now you can use the openai_api_key variable in your application logic
print(f'Your OpenAI API Key is: {openai_api_key}')


def main_logic():
    # Your main application logic here
    return "This is the result from the core application logic."
