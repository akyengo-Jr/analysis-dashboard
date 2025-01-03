import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
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

def check_null_values(data):
    null_summary = data.isnull().sum()
    st.write("Null Values Summary:")
    st.dataframe(null_summary)

def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        return data
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

def clean_data(data, cleaning_option):
    data_cleaned = data.copy()
    if cleaning_option == "Drop missing values":
        data_cleaned = data_cleaned.dropna()
    elif cleaning_option == "Fill missing values":
        fill_method = st.radio("Fill missing values with:", ["Mean", "Custom Value"])
        if fill_method == "Mean":
            data_cleaned = data_cleaned.fillna(data_cleaned.mean(numeric_only=True))
        else:
            fill_value = st.text_input("Enter value to fill missing data", "0")
            try:
                fill_value = float(fill_value)
            except ValueError:
                pass
            data_cleaned = data_cleaned.fillna(fill_value)
    return data_cleaned

def convert_columns_to_numeric(data, columns):
    for column in columns:
        data[column], _ = pd.factorize(data[column])
        data[column] = data[column].astype(float)
    return data

def save_cleaned_data(data_cleaned):
    try:
        towrite = BytesIO()
        data_cleaned.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=towrite,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )
        st.success("Cleaned dataset ready for download!")
    except Exception as e:
        logger.error(f"Error preparing the cleaned dataset for download: {e}")
        st.error(f"Error preparing the cleaned dataset for download: {e}")

# Date management functions
def filter_by_date(data, date_column, start_date, end_date):
    try:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (data[date_column] >= start_date) & (data[date_column] <= end_date)
        return data.loc[mask]
    except Exception as e:
        logger.error(f"Error filtering data by date: {e}")
        st.error(f"Error filtering data by date: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def convert_to_proper_date_type(data, date_column):
    try:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        return data
    except Exception as e:
        logger.error(f"Error converting to proper date type: {e}")
        st.error(f"Error converting to proper date type: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
# String cleaning functions
def clean_strings(data, columns, unwanted_content):
    for column in columns:
        data[column] = data[column].replace(unwanted_content, '', regex=True)
    return data

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data_cleaned = data.copy()  # Initialize data_cleaned as a copy of data

    with st.expander("Null Values Management"):
        null_management_option = st.selectbox("Choose null values management option", ["Check for Null Values", "Clean Data"])
        if null_management_option == "Check for Null Values":
            if st.button("Check for Null Values"):
                check_null_values(data)
        elif null_management_option == "Clean Data":
            cleaning_option = st.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])
            data_cleaned = clean_data(data, cleaning_option)

    with st.expander("Date Management"):
        date_management_option = st.selectbox("Choose date management option", ["Filter by Date", "Convert to Proper Date Type"])
        date_column = st.selectbox("Select date column", options=data_cleaned.columns)
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")

        if date_management_option == "Filter by Date" and st.button("Apply Date Filter"):
            data_cleaned = filter_by_date(data_cleaned, date_column, start_date, end_date)
            st.success("Data filtered by date successfully!")
        elif date_management_option == "Convert to Proper Date Type" and st.button("Convert Date Type"):
            data_cleaned = convert_to_proper_date_type(data_cleaned, date_column)
            st.success("Date type conversion successful!")

    with st.expander("String Cleaning"):
        string_columns = st.multiselect("Select columns to clean strings", data_cleaned.select_dtypes(include=['object']).columns)
        unwanted_content = st.text_area("Enter unwanted content (regex supported)")

        if st.button("Clean Strings"):
            data_cleaned = clean_strings(data_cleaned, string_columns, unwanted_content)
            st.success("Strings cleaned successfully!")

    with st.expander("Save Cleaned Dataset"):
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
