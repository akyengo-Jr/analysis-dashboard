import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import seaborn as sns

st.set_page_config(layout='wide', page_title='Data Analytics Dashboard')

st.title("Interactive Data Analytics Dashboard")

# File uploader for dynamic dataset loading
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
  try:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
  except Exception as e:
    st.sidebar.error(f"Error reading the CSV file: {e}")
    st.stop()
else:
  st.sidebar.warning("Please upload a CSV file.")
  st.stop()

# Initialize data_cleaned with the original data
data_cleaned = data.copy()

# Data cleaning options
st.sidebar.header("Data Cleaning Options")
cleaning_option = st.sidebar.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])

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

# Function to convert non-numeric columns to numeric
def convert_columns_to_numeric(data, columns):
    """
    Convert non-numeric columns to numeric by factorizing them.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    columns (list): List of column names to convert to numeric.

    Returns:
    pd.DataFrame: The dataframe with specified columns converted to numeric.
  """
    for column in columns:
        data[column], _ = pd.factorize(data[column])
        data[column] = data[column].astype(float)
    return data

# Convert non-numeric columns to numeric
st.sidebar.header("Convert Data Types")
convert_columns = st.sidebar.multiselect("Select columns to convert to numeric", data_cleaned.select_dtypes(include=['object']).columns)
if st.sidebar.button("Convert Selected Columns"):
  data_cleaned = convert_columns_to_numeric(data_cleaned, convert_columns)
  st.sidebar.success("Selected columns converted to numeric successfully!")

# Column selection for dropping
columns_to_drop = st.sidebar.multiselect("Select columns to drop", data_cleaned.columns)
if st.sidebar.button("Drop Selected Columns"):
  data_cleaned = data_cleaned.drop(columns=columns_to_drop)
  data = data_cleaned  # Update the main data variable
  st.sidebar.success("Selected columns dropped successfully!")

'''save cleaned dataset
st.sidebar.header("Save Cleaned Dataset")
cleaned_file_path = st.sidebar.text_input("Enter file path to save cleaned dataset", "/home/goodness/Notebooks/dashboardProject/cleaned_data.csv")
if st.sidebar.button("Save Cleaned Dataset"):
  try:
    data_cleaned.to_csv(cleaned_file_path, index=False)  # save the cleaned data to a csv file
    st.sidebar.success(f"Cleaned dataset saved to {cleaned_file_path}")
  except Exception as e:
    st.sidebar.error(f"Error saving the cleaned dataset: {e}")'''

st.markdown('---')
st.header("Dataset Overview")
st.dataframe(data.head())

st.subheader("Dataset Description")
st.write(data.describe())

st.header("Data Visualization")

# Apply seaborn theme
sns.set_theme(style="darkgrid")

# Bar chart example
st.subheader("Bar Chart")
x_column = st.selectbox("Select X-axis column for Bar Chart", data.columns)
y_column = st.selectbox("Select Y-axis column for Bar Chart", data.columns)
bar_chart = px.bar(data, x=x_column, y=y_column, title="Bar Chart")
st.plotly_chart(bar_chart)

# Scatter plot example
st.subheader("Scatter Plot")
x_scatter = st.selectbox("Select X-axis column for Scatter Plot", data.columns, key='scatter_x')
y_scatter = st.selectbox("Select Y-axis column for Scatter Plot", data.columns, key='scatter_y')
scatter_plot = px.scatter(data, x=x_scatter, y=y_scatter, title="Scatter Plot")
st.plotly_chart(scatter_plot)

# Line chart example
st.subheader("Line Chart")
x_line = st.selectbox("Select X-axis column for Line Chart", data.columns, key='line_x')
y_line = st.selectbox("Select Y-axis column for Line Chart", data.columns, key='line_y')
line_chart = px.line(data, x=x_line, y=y_line, title="Line Chart")
st.plotly_chart(line_chart)

# Histogram example
st.subheader("Histogram")
hist_column = st.selectbox("Select column for Histogram", data.columns, key='hist')
histogram = px.histogram(data, x=hist_column, title="Histogram")
st.plotly_chart(histogram)

# Box plot example
st.subheader("Box Plot")
x_box = st.selectbox("Select X-axis column for Box Plot", data.columns, key='box_x')
y_box = st.selectbox("Select Y-axis column for Box Plot", data.columns, key='box_y')
box_plot = px.box(data, x=x_box, y=y_box, title="Box Plot")
st.plotly_chart(box_plot)
