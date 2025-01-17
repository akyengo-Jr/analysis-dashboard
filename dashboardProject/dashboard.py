import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import seaborn as sns
import logging
from io import BytesIO



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        st.sidebar.success("File uploaded successfully!")
        return data
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        st.sidebar.error(f"Error reading the CSV file: {e}")
        st.stop()

def clean_data(data, cleaning_option):
    data_cleaned = data.copy()
    if cleaning_option == "Drop missing values":
        data_cleaned = data_cleaned.dropna()
    elif cleaning_option == "Fill missing values":
        fill_method = st.sidebar.radio("Fill missing values with:", ["Mean", "Custom Value"])
        if fill_method == "Mean":
            data_cleaned = data_cleaned.fillna(data_cleaned.mean(numeric_only=True))
        else:
            fill_value = st.sidebar.text_input("Enter value to fill missing data", "0")
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
        st.sidebar.download_button(
            label="Download Cleaned Data as CSV",
            data=towrite,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )
        st.sidebar.success("Cleaned dataset ready for download!")
    except Exception as e:
        logger.error(f"Error preparing the cleaned dataset for download: {e}")
        st.sidebar.error(f"Error preparing the cleaned dataset for download: {e}")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.header("Null Values Check")
    if st.sidebar.button("Check for Null Values"):
        check_null_values(data)
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()

st.sidebar.header("Data Cleaning Options")
cleaning_option = st.sidebar.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])
data_cleaned = clean_data(data, cleaning_option)

st.sidebar.header("Convert Data Types")
convert_columns = st.sidebar.multiselect("Select columns to convert to numeric", data_cleaned.select_dtypes(include=['object']).columns)
if st.sidebar.button("Convert Selected Columns"):
    data_cleaned = convert_columns_to_numeric(data_cleaned, convert_columns)
    st.sidebar.success("Selected columns converted to numeric successfully!")

st.sidebar.header("Save Cleaned Dataset")
if st.sidebar.button("Prepare Cleaned Dataset for Download"):
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
    st.plotly_chart(chart)

chart_types = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]
for chart_type in chart_types:
    st.subheader(chart_type)
    x_column = st.selectbox(f"Select X-axis column for {chart_type}", data.columns, key=f'{chart_type}_x')
    y_column = st.selectbox(f"Select Y-axis column for {chart_type}", data.columns, key=f'{chart_type}_y')
    plot_chart(chart_type, data, x_column, y_column)
