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

# Set page config
st.set_page_config(layout='centered', page_title='Data Analytics Dashboard', menu_items={})
st.title("Data Dashboard")

# Load CSS file for styling
with open("dashboardProject/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def check_null_values(data):
    null_summary = data.isnull().sum()
    st.write("Summary of Null Values in Each Column:")
    st.write(null_summary)
    return null_summary
def load_data(uploaded_file):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        return data
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        st.sidebar.error(f"Error reading the CSV file: {e}")
        st.stop()

def drop_columns(data, columns):
    """Drop specified columns from the dataframe."""
    return data.drop(columns=columns, errors='ignore')

def clean_data(data, cleaning_option):
    """Clean the data based on the selected cleaning option."""
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
    """Convert specified columns to numeric type."""
    for column in columns:
        data[column], _ = pd.factorize(data[column])
        data[column] = data[column].astype(float)
    return data

def save_cleaned_data(data_cleaned):
    """Save the cleaned data to a CSV file and provide a download button."""
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

def generate_summary_statistics(data):
    """Generate and display summary statistics of the dataset."""
    st.subheader("Summary Statistics")
    st.write(data.describe())

def filter_data(data, column, filter_value, filter_type='exact'):
    """Filter data based on a specified column, value, and filter type."""
    if column not in data.columns:
        logger.error(f"Column '{column}' not found in the data.")
        st.sidebar.error(f"Column '{column}' not found in the data.")
        return data

    try:
        if filter_type == 'exact':
            return data[data[column] == filter_value]
        elif filter_type == 'partial':
            return data[data[column].astype(str).str.contains(str(filter_value))]
        elif filter_type == 'range':
            if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                return data[(data[column] >= filter_value[0]) & (data[column] <= filter_value[1])]
            else:
                raise ValueError("For 'range' filter type, filter_value must be a list or tuple with two elements.")
        else:
            raise ValueError(f"Invalid filter_type: {filter_type}")
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        st.sidebar.error(f"Error filtering data: {e}")
        return data

def plot_chart(chart_type, data, x_column, y_column):
    """Plot chart based on the selected type and columns."""
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

# File uploader for CSV files
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.sidebar.warning("Please upload a CSV file.")
    st.stop()
    
# Check for null values.
st.sidebar.header("Check for Null Values")
if st.sidebar.button("Check Null Values"):
    check_null_values(data)

# Data cleaning options
st.sidebar.header("Data Cleaning Options")
cleaning_option = st.sidebar.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])
data_cleaned = clean_data(data, cleaning_option)


# Drop columns option
st.sidebar.header("Drop Columns")
drop_columns_option = st.sidebar.multiselect("Select columns to drop", data_cleaned.columns)
if st.sidebar.button("Drop Selected Columns"):
    data_cleaned = drop_columns(data_cleaned, drop_columns_option)
    st.sidebar.success("Selected columns dropped successfully!")

# Data filtering options
st.sidebar.header("Data Filtering Options")
filter_column = st.sidebar.selectbox("Select column to filter", options=data_cleaned.columns)
filter_value = st.sidebar.text_input("Enter value to filter by")
filter_type = st.sidebar.selectbox("Select filter type", ["exact", "partial", "range"])

if st.sidebar.button("Filter Data"):
    data_cleaned = filter_data(data_cleaned, filter_column, filter_value, filter_type)
    st.sidebar.success("Data filtered successfully!")

# Convert data types options
st.sidebar.header("Convert Data Types")
convert_columns = st.sidebar.multiselect("Select columns to convert to numeric", data_cleaned.select_dtypes(include=['object']).columns)
if st.sidebar.button("Convert Selected Columns"):
    data_cleaned = convert_columns_to_numeric(data_cleaned, convert_columns)
    st.sidebar.success("Selected columns converted to numeric successfully!")

# Save cleaned dataset
st.sidebar.header("Save Cleaned Dataset")
if st.sidebar.button("Prepare Cleaned Dataset for Download"):
    save_cleaned_data(data_cleaned)

# Display dataset overview and statistics
st.markdown('---')
st.header("Dataset Overview")
st.dataframe(data.head())

generate_summary_statistics(data)

# Data visualization section
st.header("Data Visualization")
sns.set_theme(style="darkgrid")

# Display chart types for visualization
chart_types = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]
for chart_type in chart_types:
    st.subheader(chart_type)
    x_column = st.selectbox(f"Select X-axis column for {chart_type}", data.columns, key=f'{chart_type}_x')
    y_column = st.selectbox(f"Select Y-axis column for {chart_type}", data.columns, key=f'{chart_type}_y')
    plot_chart(chart_type, data, x_column, y_column)
