import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import logging
from io import BytesIO
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(layout='wide', page_title='Data Analytics Dashboard')
st.title("Data Dashboard")

# Load CSS file
with open("dashboardProject/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar with expanders
with st.sidebar:
    st.header("Sidebar Menu")
    with st.expander("Upload CSV"):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    with st.expander("Null Values Management"):
        null_management_option = st.selectbox("Choose null values management option", ["Check for Null Values", "Clean Data"])
    with st.expander("Date Management"):
        date_management_option = st.selectbox("Choose date management option", ["Filter by Date", "Convert to Proper Date Type"])
    with st.expander("String Cleaning"):
        string_columns = st.multiselect("Select columns to clean strings", [])
        unwanted_content = st.text_area("Enter unwanted content (regex supported)")
    with st.expander("Save Cleaned Dataset"):
        st.button("Prepare Cleaned Dataset for Download")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data_cleaned = data.copy()  # Initialize data_cleaned as a copy of data

    if null_management_option == "Check for Null Values":
        if st.button("Check for Null Values"):
            check_null_values(data)
    elif null_management_option == "Clean Data":
        cleaning_option = st.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])
        data_cleaned = clean_data(data, cleaning_option)

    if date_management_option == "Filter by Date" and st.button("Apply Date Filter"):
        data_cleaned = filter_by_date(data_cleaned, date_column, start_date, end_date)
        st.success("Data filtered by date successfully!")
    elif date_management_option == "Convert to Proper Date Type" and st.button("Convert Date Type"):
        data_cleaned = convert_to_proper_date_type(data_cleaned, date_column)
        st.success("Date type conversion successful!")

    if st.button("Clean Strings"):
        data_cleaned = clean_strings(data_cleaned, string_columns, unwanted_content)
        st.success("Strings cleaned successfully!")

    if st.button("Prepare Cleaned Dataset for Download"):
        save_cleaned_data(data_cleaned)

    st.markdown('---')
    st.header("Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Dataset Description")
    st.write(data.describe())

    st.header("Data Visualization")
    sns.set_theme(style="darkgrid")

    def plot_chart(chart_type, data, x_column, y_column):
        if chart_type == "Bar Chart":
            chart = px.bar(data, x=x_column, y=y_column, title="Bar Chart")
        elif chart_type == "Scatter Plot":
            chart = px.scatter(data, x=x_column, y=y_column, title="Scatter Plot")
        elif chart_type == "Line Chart":
            chart = px.line(data, x=x_column, y=y_column, title="Line Chart")
        elif chart_type == "Histogram":
            chart = px.histogram(data, x=x_column, title="Histogram")
        elif chart_type == "Box Plot":
            chart = px.box(data, x=x_column, y=y_column, title="Box Plot")
        st.plotly_chart(chart, use_container_width=True)

    chart_types = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]
    for chart_type in chart_types:
        st.subheader(chart_type)
        x_column = st.selectbox(f"Select X-axis column for {chart_type}", data.columns, key=f'{chart_type}_x')
        y_column = st.selectbox(f"Select Y-axis column for {chart_type}", data.columns, key=f'{chart_type}_y')
        plot_chart(chart_type, data, x_column, y_column)
else:
    st.warning("Please upload a CSV file.")
    st.stop()
