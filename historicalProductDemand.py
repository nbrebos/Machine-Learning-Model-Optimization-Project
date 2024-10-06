#!"C:\\Anaconda\\envs\\tf\\python.exe"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load the Superstore Sales Dataset
df = pd.read_csv('Assignments_Datasets/Historical Product Demand.csv')
print(df.columns)
df = df.drop(columns=['Warehouse', 'Product_Category'])
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
print(df.info())
# Drop rows with NaN values in 'Date' or 'Order_Demand' columns
df = df.dropna(subset=['Date', 'Order_Demand'])
# Convert Order Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
# Extract Order Date Information
df['Order Month'] = df['Date'].dt.month
df['Order Quarter'] = (df['Date'].dt.month - 1) // 3 + 1
df['Order Half'] = (df['Date'].dt.month - 1) // 6 + 1
# Filter the dataset for the specified products
products_of_interest = ['Product_0973', 'Product_0979', 'Product_1970']


# Initialize DataFrame to store predictions by quarter, half-year, and month
predictions_by_quarter = pd.DataFrame(columns=['Order Quarter', 'Order_Demand', 'Predicted Orders (Quarter)'])
predictions_by_half = pd.DataFrame(columns=['Order Half', 'Order_Demand', 'Predicted Orders (Half)'])
predictions_by_month = pd.DataFrame(columns=['Order Month', 'Order_Demand', 'Predicted Orders (Month)'])



# Iterate over each product of interest
for product_code in products_of_interest:
    # Filter the data for the current product
    df_product = df[df['Product_Code'] == product_code]
    # Aggregate order demand for each quarter, half-year, and month
    sum_demand_quarter = df_product.groupby('Order Quarter')['Order_Demand'].sum().reset_index()
    sum_demand_quarter.columns = ['Order Quarter', 'Sum Demand Quarter']
    # Convert 'Order Quarter' column to int64 in both dataframes
    df_product['Order Quarter'] = df_product['Order Quarter'].astype('int64')
    sum_demand_quarter['Order Quarter'] = sum_demand_quarter['Order Quarter'].astype('int64')
    sum_demand_half = df_product.groupby('Order Half')['Order_Demand'].sum().reset_index()
    sum_demand_half.columns = ['Order Half', 'Sum Demand Half']
    # Convert 'Order Half' column to int64 in both dataframes
    df_product['Order Half'] = df_product['Order Half'].astype('int64')
    sum_demand_half['Order Half'] = sum_demand_half['Order Half'].astype('int64')
    sum_demand_month = df_product.groupby('Order Month')['Order_Demand'].sum().reset_index()
    sum_demand_month.columns = ['Order Month', 'Sum Demand Month']
    # Merge sum of demand for each quarter, half-year, and month back into the original dataset
    df_product = pd.merge(df_product, sum_demand_quarter, on='Order Quarter', how='left')
    df_product = pd.merge(df_product, sum_demand_half, on='Order Half', how='left')
    df_product = pd.merge(df_product, sum_demand_month, on='Order Month', how='left')
    # Define features and target variable
    features_q = ['Order Quarter', 'Sum Demand Quarter']
    features_h = ['Order Half', 'Sum Demand Half']
    features_m = ['Order Month', 'Sum Demand Month']
    target = 'Order_Demand'

    # Filter rows before 2019 for training and rows from 2019 for prediction
    train_df= df_product[df_product['Date'].dt.year < 2016]
    test_df = df_product[df_product['Date'].dt.year == 2016]

    # Create a pipeline for preprocessing
    preprocessor_q = ColumnTransformer(
        transformers=[
            ('categorical_q', OneHotEncoder(handle_unknown='ignore'), ['Order Quarter']),  # Encode categorical columns for quarter
        ],
        remainder='passthrough'  # Pass through the remaining columns unchanged
    )
    preprocessor_h = ColumnTransformer(
        transformers=[
            ('categorical_h', OneHotEncoder(handle_unknown='ignore'), ['Order Half']),  # Encode categorical columns for half-year
        ],
        remainder='passthrough'  # Pass through the remaining columns unchanged
    )
    preprocessor_m = ColumnTransformer(
        transformers=[
            ('categorical_m', OneHotEncoder(handle_unknown='ignore'), ['Order Month'])  # Encode categorical columns for month
        ],
        remainder='passthrough'  # Pass through the remaining columns unchanged
    )
    # Combine preprocessing with regression model
    model_q = Pipeline([
        ('preprocessor', preprocessor_q),
        ('regressor', LinearRegression())
    ])
    model_h = Pipeline([
        ('preprocessor', preprocessor_h),
        ('regressor', LinearRegression())
    ])
    model_m = Pipeline([
        ('preprocessor', preprocessor_m),
        ('regressor', LinearRegression())
    ])


    # Train the regression model for each quarter
    model_q.fit(train_df[features_q], train_df[target])
    # Train the regression model for each half-year
    model_h.fit(train_df[features_h], train_df[target])
    # Train the regression model for each month
    model_m.fit(train_df[features_m], train_df[target])
    # Predict orders for each quarter, half-year, and month on the test set
    predictions_q = model_q.predict(test_df[features_q])
    predictions_h = model_h.predict(test_df[features_h])
    predictions_m = model_m.predict(test_df[features_m])
     # Create DataFrame to store predictions for each quarter, half-year, and month
    predictions_by_quarter = pd.concat([predictions_by_quarter, pd.DataFrame({'Order Date': test_df['Date'],
                                                                               'Predicted Orders (Quarter)': predictions_q})], ignore_index=True)
    predictions_by_half = pd.concat([predictions_by_half, pd.DataFrame({'Order Date': test_df['Date'],
                                                                         'Predicted Orders (Half)': predictions_h})], ignore_index=True)
    predictions_by_month = pd.concat([predictions_by_month, pd.DataFrame({'Order Date': test_df['Date'],
                                                                           'Predicted Orders (Month)': predictions_m})], ignore_index=True)
    # Ensure 'Order Date' column is datetime
    predictions_by_quarter['Order Date'] = pd.to_datetime(predictions_by_quarter['Order Date'])
    predictions_by_half['Order Date'] = pd.to_datetime(predictions_by_half['Order Date'])
    predictions_by_month['Order Date'] = pd.to_datetime(predictions_by_month['Order Date'])
   # Evaluate the predictions
    mse_q = mean_squared_error(test_df['Order_Demand'], predictions_q)
    mae_q = mean_absolute_error(test_df['Order_Demand'], predictions_q)
    mse_h = mean_squared_error(test_df['Order_Demand'], predictions_h)
    mae_h = mean_absolute_error(test_df['Order_Demand'], predictions_h)
    mse_m = mean_squared_error(test_df['Order_Demand'], predictions_m)
    mae_m = mean_absolute_error(test_df['Order_Demand'], predictions_m)
    # Define the filename for the metrics
    filename_metrics =  'question2/metrics_product_{}.txt'.format(product_code)
            # Create a string representation of the metrics
    metrics_string = "Mean Squared Error (MSE) for quarter predictions: {}\n".format(mse_q)
    metrics_string += "Mean Absolute Error (MAE) for quarter predictions: {}\n".format(mae_q)
    metrics_string += "Mean Squared Error (MSE) for half-year predictions: {}\n".format(mse_h)
    metrics_string += "Mean Absolute Error (MAE) for half-year predictions: {}\n".format(mae_h)
    metrics_string += "Mean Squared Error (MSE) for month predictions: {}\n".format(mse_m)
    metrics_string += "Mean Absolute Error (MAE) for month predictions: {}\n".format(mae_m)
        # Save the metrics to a text file
    with open(filename_metrics, 'w') as file:
        file.write(metrics_string)
    # Print the metrics
    print(metrics_string)
    # Save the metrics plot as an image
    plt.figure()
    plt.text(0.5, 0.5, metrics_string, ha='center', va='center')
    plt.axis('off')
    plt.savefig('question2/metrics_product_{}.png'.format(product_code), bbox_inches='tight')
    plt.close()
    # Plot line chart for sum of predicted orders and sum of actual orders per quarter
    plt.figure(figsize=(10, 6))
    plt.plot(test_df.groupby(test_df['Date'].dt.quarter)['Order_Demand'].sum(), marker='o', linestyle='-', label='Sum Actual Orders')
    plt.plot(predictions_by_quarter.groupby(predictions_by_quarter['Order Date'].dt.quarter)['Predicted Orders (Quarter)'].sum(),
              marker='o', linestyle='-', label='Sum Predicted Orders (Quarter)')
    plt.title('Sum Predicted vs. Actual Orders per Quarter for Product {}: MSE={:.2f}, MAE={:.2f}'.format(product_code, mse_q, mae_q))
    plt.xlabel('Quarter')
    plt.ylabel('Orders')
    plt.legend()
    plt.grid(True)
    # Save the plot with a custom name
    plt.savefig('question2/predicted_vs_actual_orders_quarter_{}.png'.format(product_code))
    plt.show()
    # Plot line chart for sum of predicted orders and sum of actual orders per half-year
    plt.figure(figsize=(10, 6))
    plt.plot(test_df.groupby(test_df['Date'].dt.month // 6)['Order_Demand'].sum(), marker='o', linestyle='-', label='Sum Actual Orders')
    plt.plot(predictions_by_half.groupby(predictions_by_half['Order Date'].dt.month // 6)['Predicted Orders (Half)'].sum(),
              marker='o', linestyle='-', label='Sum Predicted Orders (Half)')
    plt.title('Sum Predicted vs. Actual Orders per Half-year for Product {}: MSE={:.2f}, MAE={:.2f}'.format(product_code, mse_h, mae_h))
    plt.xlabel('Half-year')
    plt.ylabel('Orders')
    plt.legend()
    plt.grid(True)
    plt.savefig('question2/predicted_vs_actual_orders_halfyear_{}.png'.format(product_code))
    plt.show()
    # Plot line chart for sum of predicted orders and sum of actual orders per month
    plt.figure(figsize=(10, 6))
    plt.plot(test_df.groupby(test_df['Date'].dt.month)['Order_Demand'].sum(), marker='o', linestyle='-', label='Sum Actual Orders')
    plt.plot(predictions_by_month.groupby(predictions_by_month['Order Date'].dt.month)['Predicted Orders (Month)'].sum(),
              marker='o', linestyle='-', label='Sum Predicted Orders (Month)')
    plt.title('Sum Predicted vs. Actual Orders per Month for Product {}: MSE={:.2f}, MAE={:.2f}'.format(product_code, mse_m, mae_m))
    plt.xlabel('Month')
    plt.ylabel('Orders')
    plt.legend()
    plt.grid(True)
    plt.savefig('question2/predicted_vs_actual_orders_month_{}.png'.format(product_code))
    plt.show()
