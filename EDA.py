#!"C:\\Anaconda\\envs\\tf\\python.exe"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the Superstore Sales Dataset
df = pd.read_csv('Assignments_Datasets/superstore_sales.csv')
df.drop(['Row ID'], axis=1, inplace=True)


# Check for missing or infinite values
missing_values = df.isnull().sum()
infinite_values = df.isin([np.inf, -np.inf]).sum()

print("Missing Values:")
print(missing_values)
print("\nInfinite Values:")
print(infinite_values)
df.dropna(subset=['Postal Code'], inplace=True)



# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

# Extract Order Date Information
df['Order Month'] = df['Order Date'].dt.month
df['Order Quarter'] = (df['Order Date'].dt.month - 1) // 3 + 1
df['Order Half'] = (df['Order Date'].dt.month - 1) // 6 + 1

# Aggregate sales for each period and add as additional columns
sum_sales_month = df.groupby('Order Month')['Sales'].sum().reset_index()
sum_sales_month.columns = ['Order Month', 'Sum Sales Month']
sum_sales_quarter = df.groupby('Order Quarter')['Sales'].sum().reset_index()
sum_sales_quarter.columns = ['Order Quarter', 'Sum Sales Quarter']
sum_sales_half = df.groupby('Order Half')['Sales'].sum().reset_index()
sum_sales_half.columns = ['Order Half', 'Sum Sales Half']

# Merge sum of sales for each period back into the original dataset
df = pd.merge(df, sum_sales_month, on='Order Month', how='left')
df = pd.merge(df, sum_sales_quarter, on='Order Quarter', how='left')
df = pd.merge(df, sum_sales_half, on='Order Half', how='left')

pd.set_option('display.max_columns', None)
print(df.head())


# Pie Chart of Ship Mode
plt.figure(figsize=(8, 8))
ship_mode_counts = df['Ship Mode'].value_counts()
plt.pie(ship_mode_counts, labels=ship_mode_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Orders by Ship Mode')
plt.show()


# Next, we can create a bar plot to visualize sales by category
plt.figure(figsize=(10, 6))
sns.barplot(x='Sub-Category', y='Sales', data=df, ci=None)
plt.title('Sales by Sub-Category')
plt.xlabel('Sub-Category')
plt.ylabel('Sales')
# Rotate x-axis labels for better readability
plt.xticks(rotation=70)
plt.show()

# Resample data by date and count the number of orders for each month
monthly_order_frequency = df.groupby(pd.Grouper(key='Order Date', freq='M')).size()

# Line Chart of Order Frequency Over Time (Aggregated by Month)
plt.figure(figsize=(12, 6))  # Increase the size of the plot
plt.plot(monthly_order_frequency.index, monthly_order_frequency.values, linestyle='-')
plt.title('Monthly Order Frequency')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.xticks(monthly_order_frequency.index, monthly_order_frequency.index.strftime('%b %Y'), rotation=45)  # Specify the x-axis ticks with month names
plt.grid(True)
plt.show()


# For Order Month
plt.figure(figsize=(10, 6))
plt.plot(sum_sales_month['Order Month'], sum_sales_month['Sum Sales Month'], marker='o', linestyle='-')
plt.title('Sales Over Time (Monthly)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# For Order Quarter
plt.figure(figsize=(10, 6))
plt.plot(sum_sales_quarter['Order Quarter'], sum_sales_quarter['Sum Sales Quarter'], marker='o', linestyle='-')
plt.title('Sales Over Time (Quarterly)')
plt.xlabel('Quarter')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# For Order Half
plt.figure(figsize=(10, 6))
plt.plot(sum_sales_half['Order Half'], sum_sales_half['Sum Sales Half'], marker='o', linestyle='-')
plt.title('Sales Over Time (Half-Yearly)')
plt.xlabel('Half')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# Pie Chart of Category Distribution
plt.figure(figsize=(8, 8))
df_category = df.groupby('Category')['Sales'].sum().reset_index()
plt.pie(df_category['Sales'], labels=df_category['Category'], autopct='%1.1f%%')
plt.title('Category Distribution of Sales')
plt.show()

# Select the numerical columns for the pairplot
numerical_columns = ['Sales', 'Order Month', 'Order Quarter', 'Order Half', 'Sum Sales Month', 'Sum Sales Quarter', 'Sum Sales Half']

# Create a pairplot
#sns.pairplot(df[numerical_columns])
#plt.show()
# Exclude non-numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])
print(numeric_df)

# Compute correlation matrix
corr = numeric_df.corr()
# Convert correlation matrix DataFrame to NumPy array
corr_array = corr.to_numpy()
def plot_heatmap_with_counts(matrix, variable_names):
    plt.figure(figsize=(12, 7))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', linewidths=.5,
                xticklabels=variable_names, yticklabels=variable_names)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
    plt.title('Heatmap with Annotated Counts')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.show()


plot_heatmap_with_counts(corr_array,numeric_df.columns)

# Filter rows before 2019 for training and rows from 2019 for prediction
train_data = df[df['Order Date'].dt.year < 2018]
test_data = df[df['Order Date'].dt.year == 2018]


print(test_data.head())


# Define features and target variable
features = ['Order Month', 'Order Quarter', 'Order Half','City','Category','Sub-Category','Sum Sales Month', 'Sum Sales Quarter', 'Sum Sales Half' ]
#features = ['Order Month','Order Quarter','Order Half','City','Category','Sub-Category','Product Name','Sum Sales Month' ]
target = 'Sales'
# 'Sum Sales Month', 'Sum Sales Quarter', 'Sum Sales Half' ,

# Create a pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'),['Order Month','Order Quarter', 'Order Half','City','Category','Sub-Category'])  # Encode categorical columns
    ],
    remainder='passthrough'  # Pass through the remaining columns unchanged
)

# Combine preprocessing with regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the regression model
model.fit(train_data[features], train_data[target])

# Predict sales for the year 2018
test_data['Predicted Sales'] = model.predict(test_data[features])

# Initialize a DataFrame to store predictions by month
predictions_by_month = pd.DataFrame(columns=['Order Date', 'Sales', 'Predicted Sales'])
# Loop through each month in the test data
for month in range(1, 13):
    # Filter data for the current month
    test_month_data = test_data[test_data['Order Month'] == month]

    # Predict sales for the current month
    test_month_data['Predicted Sales'] = model.predict(test_month_data[features])

    # Append predictions for the current month to the DataFrame
    predictions_by_month = pd.concat([predictions_by_month, test_month_data[['Order Date', 'Sales', 'Predicted Sales']]], ignore_index=True)



# Ensure 'Order Month' column is datetime
predictions_by_month['Order Month'] = predictions_by_month['Order Date'].dt.month

# Calculate sum of predicted sales for each month
sum_predicted_sales = predictions_by_month.groupby('Order Month')['Predicted Sales'].sum()

# Calculate sum of actual sales for each month
sum_actual_sales = predictions_by_month.groupby('Order Month')['Sales'].sum()



# Plot line chart for sum of predicted sales and sum of actual sales per month
plt.figure(figsize=(10, 6))
plt.plot(sum_predicted_sales.index, sum_predicted_sales.values, marker='o', linestyle='-', label='Sum Predicted Sales')
plt.plot(sum_actual_sales.index, sum_actual_sales.values, marker='o', linestyle='-', label='Sum Actual Sales')
plt.title('Sum Predicted vs. Actual Sales per Month')
plt.xlabel('Order Month')
plt.ylabel('Sales')
plt.xticks(range(1, 13))  # Set x-axis ticks to represent months
plt.legend()
plt.grid(True)
plt.show()

# Calculate MAE
mae = mean_absolute_error(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print("Mean Absolute Error (MAE):", mae)
# Calculate MSE
mse = mean_squared_error(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print("Mean Squared Error (MSE):", mse)
r2 = r2_score(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print(f"R-squared (R2): {r2}")



# Compute VIF for each feature
vif_data = numeric_df.drop(columns=['Sales'])  # Exclude the target variable
vif_values = pd.DataFrame()
vif_values["Feature"] = vif_data.columns
vif_values["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("Variance Inflation Factors:")
print(vif_values)



# Define features and target variable
features = ['Order Month', 'City','Category','Sub-Category','Product Name','Sum Sales Month' ]
#features = ['Order Month','Order Quarter','Order Half','City','Category','Sub-Category','Product Name','Sum Sales Month' ]
target = 'Sales'
# 'Sum Sales Month', 'Sum Sales Quarter', 'Sum Sales Half' ,

# Create a pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'),['Order Month', 'City','Category','Sub-Category','Product Name'])  # Encode categorical columns
    ],
    remainder='passthrough'  # Pass through the remaining columns unchanged
)

# Combine preprocessing with regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the regression model
model.fit(train_data[features], train_data[target])

test_data = df[df['Order Date'].dt.year == 2018]
# Predict sales for the year 2018
test_data['Predicted Sales'] = model.predict(test_data[features])

# Initialize a DataFrame to store predictions by month
predictions_by_month = pd.DataFrame(columns=['Order Date', 'Sales', 'Predicted Sales'])
# Loop through each month in the test data
for month in range(1, 13):
    # Filter data for the current month
    test_month_data = test_data[test_data['Order Month'] == month]

    # Predict sales for the current month
    test_month_data['Predicted Sales'] = model.predict(test_month_data[features])

    # Append predictions for the current month to the DataFrame
    predictions_by_month = pd.concat([predictions_by_month, test_month_data[['Order Date', 'Sales', 'Predicted Sales']]], ignore_index=True)



# Ensure 'Order Month' column is datetime
predictions_by_month['Order Month'] = predictions_by_month['Order Date'].dt.month

# Calculate sum of predicted sales for each month
sum_predicted_sales = predictions_by_month.groupby('Order Month')['Predicted Sales'].sum()

# Calculate sum of actual sales for each month
sum_actual_sales = predictions_by_month.groupby('Order Month')['Sales'].sum()



# Plot line chart for sum of predicted sales and sum of actual sales per month
plt.figure(figsize=(10, 6))
plt.plot(sum_predicted_sales.index, sum_predicted_sales.values, marker='o', linestyle='-', label='Sum Predicted Sales')
plt.plot(sum_actual_sales.index, sum_actual_sales.values, marker='o', linestyle='-', label='Sum Actual Sales')
plt.title('Sum Predicted vs. Actual Sales per Month')
plt.xlabel('Order Month')
plt.ylabel('Sales')
plt.xticks(range(1, 13))  # Set x-axis ticks to represent months
plt.legend()
plt.grid(True)
plt.show()

# Calculate MAE
mae = mean_absolute_error(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print("Mean Absolute Error (MAE):", mae)
# Calculate MSE
mse = mean_squared_error(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print("Mean Squared Error (MSE):", mse)
r2 = r2_score(predictions_by_month['Sales'], predictions_by_month['Predicted Sales'])
print(f"R-squared (R2): {r2}")



