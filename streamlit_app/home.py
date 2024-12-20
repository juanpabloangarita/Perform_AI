# Perform_AI.streamlit_app.home.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from config import setup_paths # TODO: decide to erase it or to implement it
from src.main import main
from dashboard_plot import plot_dashboard
from params import CLOUD_ON, CODE_PROMO
from src.data_loader.files_extracting import FileLoader
from src.user_data import UserManager



st.set_page_config(page_title="Perform AI", page_icon="🌞", layout="wide", initial_sidebar_state="expanded")

# Capture the 'main' argument if provided
main_arg = None
if len(sys.argv) > 1:
    main_arg = sys.argv[1]
if CLOUD_ON == 'yes': # THIS IS ONLY VALID, THE PURPOSE OF THIS, FOR WHEN I AM LAUNCHING IT FROM STREAMLIT.IO APP, OTHERWISE THIS DOESN'T MAKE ANY SENSE BECAUSE main with CLOUD_ON from terminal works
    main_arg = 'main' #NOTE: REMEMBER CLOUD_ON THIS IS TEMPORAL ONCE IT IS LAUNCHED ONE TIME IT WILL GO ON FOREVER

# Initialize session state for authentication and username
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Function to display the login and sign-up form
def show_login_form():
    st.subheader('Login / Sign Up')
    # Option to switch between login and sign up
    option = st.radio("Select Option", ("Login", "Sign Up"))
    with st.container(border=True):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        user = UserManager(username=username, password=password)
        if option == "Sign Up":
            secret_code = st.text_input('Secret Code', type='password')
            if st.button('Sign Up'):
                if secret_code == CODE_PROMO:
                    if not user.user_exists:
                        user.create_user_data()
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['user_data']= user.user_data
                        response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                        st.success(f"Sign up successful! {response_main}")
                    else:
                        st.error('Username already exists.')
                else:
                    st.error('Invalid secret code.')
        else:  # Login
            if st.button('Login'):
                user.load_user_data()
                if user.user_data is not None:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['user_data']= user.user_data
                    response_main = main(st.session_state['user_data'], main_arg=str(main_arg))
                    st.success(f"Login successful! {response_main}")
                else:
                    st.session_state['authenticated'] = False
                    st.error('Invalid username or password')


# Check if user is authenticated
if not st.session_state['authenticated']:
    show_login_form()

    if st.button('GAME ON!'):
        pass # NOTE: weird behaviour, the way streamlit manages the state, it seems that as soon as this button is clicked, nothing below this line is read
else:
    # Display main app content if authenticated
    st.title('PERFORM_AI')
    st.markdown("<h3 style='margin-top: -20px;'>Your Health AI Assistant</h3>", unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        # Display user information in one row
        st.write(f'### Welcome {st.session_state["username"]}!')

        # Create a row with four columns
        col1, col2, col3, col4 = st.columns(4)

        # Display the information in each column
        with col1:
            st.write(f"**Sports Scheduled Today:** Yes.")

        with col2:
            st.write(f"**Goal:** {st.session_state['user_data']['goal']}")

        with col3:
            st.write(f"**Maintenance Calories:** {st.session_state['user_data']['passive_calories']} kcal")
        with col4:
            st.write(f"**Meals Pending:** {st.session_state['user_data']['meal'] if 'meal' in st.session_state['user_data'] else ''}") # TODO: how to make it appear quickly, without having to navigate to the nutrition page


    st.write("")
    st.write("### Performance Training")
    if st.checkbox("Show Dashboard"):
        tss_df, atl_df, ctl_df, tsb_df  = FileLoader().load_dfs(name_s=['tss', 'atl', 'ctl', 'tsb'], file_path=None, index=0)

        # Ensure index is in datetime format
        tss_df.index = pd.to_datetime(tss_df.index)

        st.write("##### Select Date Range to Filter Dashboard")

        # Date selection widgets with default values
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=tss_df.index.min().date(),  # Default to the earliest date in tss_df
                min_value=tss_df.index.min().date(),
                max_value=tss_df.index.max().date()
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=tss_df.index.max().date(),  # Default to the latest date in tss_df
                min_value=start_date,  # The minimum value for end_date is start_date
                max_value=tss_df.index.max().date()
            )

        # Convert the selected dates back to the '%Y-%m-%d' format for plot_dashboard
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Pass selected start_date and end_date (in string format) to the plot_dashboard function
        fig = plot_dashboard(tss_df, atl_df, ctl_df, tsb_df, start_date_str, end_date_str)

        # Set figure size by updating the layout
        fig.update_layout(
            autosize=True,
            width=1200,  # Adjust width as needed
            height=600   # Adjust height as needed
        )

        # Plot the dashboard
        st.plotly_chart(fig, use_container_width=True)

    if st.checkbox('Show Data'):
        final_df = FileLoader().update_final_df('home')
        st.write(final_df)
