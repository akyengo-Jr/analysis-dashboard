import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import streamlit as st

st.set_page_config(layout='wide', page_title='Data Analytics Dashboard')

st.title("Interactive Data Analytics Dashboard")

# File uploader for dynamic dataset loading
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  st.sidebar.success("File uploaded successfully!")
else:
  st.sidebar.warning("Please upload a CSV file.")
  st.stop()


# Data cleaning options
st.sidebar.header("Data Cleaning Options")
cleaning_option = st.sidebar.selectbox("Choose cleaning method", ["Drop missing values", "Fill missing values"])

if cleaning_option == "Drop missing values":
  data.dropna(inplace=True)
elif cleaning_option == "Fill missing values":
  fill_method = st.sidebar.radio("Fill missing values with:", ["Mean", "Custom Value"])
  if fill_method == "Mean":
    data.fillna(data.mean(numeric_only=True), inplace=True)
  else:
    fill_value = st.sidebar.text_input("Enter value to fill missing data", "0")
    data.fillna(fill_value, inplace=True)

# Column selection for dropping
st.sidebar.header("Drop Columns")
columns_to_drop = st.sidebar.multiselect("Select columns to drop", data.columns)
if st.sidebar.button("Drop Selected Columns"):
  data.drop(columns=columns_to_drop, inplace=True)
  st.sidebar.success("Selected columns dropped successfully!")

# Save cleaned dataset
if st.sidebar.button("Save Cleaned Dataset"):
  cleaned_file_path = "/home/goodness/Notebooks/dashboardProject/cleaned_data.csv"
  data.to_csv(cleaned_file_path, index=False)
  st.sidebar.success(f"Cleaned dataset saved to {cleaned_file_path}")

st.markdown('---')
st.header("Dataset Overview")
st.dataframe(data.head())

st.subheader("Dataset Description")
st.write(data.describe())

# Visualization (using Plotly Express):
st.markdown('---')
st.header("Visualizations")

# Bar chart example
st.subheader("Bar Chart")
x_column = st.selectbox("Select X-axis column for Bar Chart", data.columns)
y_column = st.selectbox("Select Y-axis column for Bar Chart", data.columns)
bar_chart = px.bar(data, x=x_column, y=y_column)
st.plotly_chart(bar_chart)

# Scatter plot example
st.subheader("Scatter Plot")
x_scatter = st.selectbox("Select X-axis column for Scatter Plot", data.columns, key='scatter_x')
y_scatter = st.selectbox("Select Y-axis column for Scatter Plot", data.columns, key='scatter_y')
scatter_chart = px.scatter(data, x=x_scatter, y=y_scatter, title="Scatter Plot")
st.plotly_chart(scatter_chart)

# Line chart example
st.subheader("Line Chart")
x_line = st.selectbox("Select X-axis column for Line Chart", data.columns, key='line_x')
y_line = st.selectbox("Select Y-axis column for Line Chart", data.columns, key='line_y')
line_chart = px.line(data, x=x_line, y=y_line, title="Line Chart")
st.plotly_chart(line_chart)

# Histogram example
st.subheader("Histogram")
hist_column = st.selectbox("Select column for Histogram", data.columns, key='hist')
hist_chart = px.histogram(data, x=hist_column, title="Histogram")
st.plotly_chart(hist_chart)

# Box plot example
st.subheader("Box Plot")
x_box = st.selectbox("Select X-axis column for Box Plot", data.columns, key='box_x')
y_box = st.selectbox("Select Y-axis column for Box Plot", data.columns, key='box_y')
box_chart = px.box(data, x=x_box, y=y_box, title="Box Plot")
st.plotly_chart(box_chart)

# Predictive Modeling (example)
st.markdown('---')
st.header("Predictive Modeling")

model_options = ["Linear Regression", "Random Forest", "Support Vector Machine"]
selected_model = st.selectbox("Select Model", model_options)

features = st.multiselect("Select features for prediction", data.columns)
target = st.selectbox("Select target variable", data.columns)

if features and target:
  X = data[features]
  y = data[target]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  if selected_model == "Linear Regression":
    model = LinearRegression()
  elif selected_model == "Random Forest":
    model = RandomForestRegressor()
  elif selected_model == "Support Vector Machine":
    model = SVR()

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  st.write("Predictions:", y_pred)  # Display predictions

  # Evaluation (example)
  mse = mean_squared_error(y_test, y_pred)
  st.write("Mean Squared Error:", mse)
else:
  st.warning("Please select features and target variable for prediction.")

# Unsupervised Learning (example)
st.markdown('---')
st.header("Unsupervised Learning")

clustering_options = ["K-Means"]
selected_clustering = st.selectbox("Select Clustering Algorithm", clustering_options)

clustering_features = st.multiselect("Select features for clustering", data.columns)

if clustering_features:
  X_clustering = data[clustering_features]

  if selected_clustering == "K-Means":
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(X_clustering)
    st.write("Cluster Centers:", kmeans.cluster_centers_)

    # Visualization of clusters
    cluster_chart = px.scatter(data, x=clustering_features[0], y=clustering_features[1], color='Cluster', title="K-Means Clustering")
    st.plotly_chart(cluster_chart)
else:
  st.warning("Please select features for clustering.")

# Sidebar for interactive elements
st.sidebar.title("Interactive Elements")
st.sidebar.markdown('Click on "Explore Your Data" to see the first few rows and interact with the chart.')

if st.sidebar.button("Load Data"):
  st.plotly_chart(px.bar(data, x=x_column, y=y_column))  # Display bar chart
