import streamlit as st
from datetime import datetime, timedelta


# Helper function to get the start of the week (Monday) for a given date
def get_monday(d: datetime):
    return d - timedelta(days=d.weekday())

# Function to get the list of dates for the week (Monday to Sunday)
def get_week_dates(week_start):
    return [week_start + timedelta(days=i) for i in range(7)]

# Function to highlight the current day
def highlight_today(week_dates):
    today = datetime.now().date()
    return [date.date() == today for date in week_dates]

# Function to display the weekly agenda using Streamlit components
def display_week(week_dates, today_highlight):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    cols = st.columns(7, gap="small")

    for i, col in enumerate(cols):
        with col:
            with st.container():
                # Highlight today's date by changing background color
                if today_highlight[i]:
                    st.markdown(f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px;'>"
                                f"<b>{days[i]}</b><br><span style='font-size: 20px;'>{week_dates[i].day}</span>"
                                f"</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='padding:10px;'>"
                                f"<b>{days[i]}</b><br><span style='font-size: 20px;'>{week_dates[i].day}</span>"
                                f"</div>", unsafe_allow_html=True)

# Main function to handle the calendar view
def main():
    st.set_page_config(layout="wide")

    # Set the default date to today
    current_date = datetime.now()

    # Use Streamlit session state to track the currently displayed week
    if 'week_start' not in st.session_state:
        st.session_state.week_start = get_monday(current_date)

    # Create the header with 'Today', '<', '>', and the month/year on the same row, aligned to the left
    col1, _, col3, col4, col5 = st.columns(5, gap="small")  # Left aligned using columns

    with col1:
        st.markdown("<h1 style='margin-bottom: 0;'>Calendar</h1>", unsafe_allow_html=True)
        st.write("")

    # Today Button
    with col3:
        if st.button("Today"):
            st.session_state.week_start = get_monday(current_date)

    # Navigation Buttons (< and >)
    with col4:
        prev_week, next_week = st.columns([1, 1], gap="small")

        with prev_week:
            if st.button("<"):
                st.session_state.week_start -= timedelta(weeks=1)

        with next_week:
            if st.button("\>"):
                st.session_state.week_start += timedelta(weeks=1)

    # Current Month and Year
    with col5:
        current_month_year = st.session_state.week_start.strftime("%B %Y")
        st.markdown(f"<h4 style='text-align:left; margin-bottom: 0;'>{current_month_year}</h4>", unsafe_allow_html=True)


    # Get the current week's dates (Monday to Sunday)
    week_dates = get_week_dates(st.session_state.week_start)

    # Determine which day is today to highlight
    today_highlight = highlight_today(week_dates)

    # Display the week view using columns for each day
    st.divider()  # Separate navigation from the calendar
    display_week(week_dates, today_highlight)


main()
