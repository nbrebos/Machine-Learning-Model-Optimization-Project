#!"C:\\Anaconda\\envs\\tf\\python.exe"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.metrics import auc

# Load the dataset
df = pd.read_csv('Assignments_Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)
 
 #Perform EDA
print(df.head())
print(df.info())
print(df.describe())
# Identify potential predictors based on correlation analysis
# For example, let's choose 'SeniorCitizen', 'tenure', 'MonthlyCharges', and 'TotalCharges'
# Convert 'TotalCharges' column to numeric, handling errors and stripping whitespace
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

# Check if there are any missing values after conversion
print("Missing values in TotalCharges column after conversion:", df['TotalCharges'].isnull().sum())

# Handle missing values
df.dropna(inplace=True)

# Check for zeros
zero_counts = (df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']] == 0).sum()
print("\nNumber of zeros in the selected columns:")
print(zero_counts)
# Check if there are any missing values after conversion
print("Missing values in TotalCharges column after conversion:", df['TotalCharges'].isnull().sum())

# Handle missing values
# Convert numeric columns to numeric data types
numeric_df = df[['SeniorCitizen', 'tenure', 'MonthlyCharges']].apply(pd.to_numeric, errors='coerce')

# Check for non-numeric values
invalid_values = df[numeric_df.isnull().any(axis=1)]
print("Rows with non-numeric values:")
print(invalid_values)
# Drop rows with non-numeric values
df.drop(invalid_values.index, inplace=True)
df.dropna(inplace=True)

# Check for missing values
missing_values = df[['SeniorCitizen', 'tenure', 'MonthlyCharges']].isnull().sum()
print("Missing values in the selected columns:")
print(missing_values)

# Check for zeros
zero_counts = (df[['SeniorCitizen', 'tenure', 'MonthlyCharges']] == 0).sum()
print("\nNumber of zeros in the selected columns:")
print(zero_counts)

# Visualize the correlation matrix for numeric columns only
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
corr_matrix = numeric_df.corr()# Define a function to plot heatmap with annotated counts for correlation matrix
def plot_heatmap_with_counts(matrix):
    plt.figure(figsize=(12, 7))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5,
                annot_kws={"color": "black"})
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            plt.text(j + 0.5, i + 0.5, f"{matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
    plt.title('Correlation Matrix with Annotated Counts')
    plt.show()

#Visualize the correlation matrix with annotated counts
plot_heatmap_with_counts(corr_matrix)

#Visualize the distribution of gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='gender')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of SeniorCitizen
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='SeniorCitizen')
plt.title('Distribution of Senior Citizens')
plt.xlabel('Senior Citizen')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of Partner
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Partner')
plt.title('Distribution of Customers with/without Partners')
plt.xlabel('Partner')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of Dependents
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Dependents')
plt.title('Distribution of Customers with/without Dependents')
plt.xlabel('Dependents')
plt.ylabel('Count')
plt.show()

#Create a histogram for tenure
plt.figure(figsize=(10, 6))
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title('Distribution of Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.show()

#Visualize the distribution of PhoneService
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='PhoneService')
plt.title('Distribution of Phone Service')
plt.xlabel('Phone Service')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of MultipleLines
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='MultipleLines')
plt.title('Distribution of Multiple Lines')
plt.xlabel('Multiple Lines')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of InternetService
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='InternetService')
plt.title('Distribution of Internet Service')
plt.xlabel('Internet Service')
plt.ylabel('Count')
plt.show()

#Visualize the distribution of OnlineSecurity
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='OnlineSecurity')
plt.title('Distribution of Online Security')
plt.xlabel('Online Security')
plt.ylabel('Count')
plt.show()

# Repeat the same process for OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, and Churn.



# Preprocess the data
# Encode categorical variables
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = label_encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = label_encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = label_encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = label_encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = label_encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = label_encoder.fit_transform(df['StreamingMovies'])
df['Contract'] = label_encoder.fit_transform(df['Contract'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = label_encoder.fit_transform(df['PaymentMethod'])
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Proceed with splitting the data and fitting the model

# Split the data into training and testing sets
X = df[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def plot_heatmap_with_counts(matrix):
    plt.figure(figsize=(12, 7))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5,
                annot_kws={"color": "black"})
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            plt.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
    plt.title('Confusion Matrix with Annotated Counts')
    plt.show()

# Plot confusion matrix
plot_heatmap_with_counts(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC and AUC
y_test_pred_lr = model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_test_pred_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC curve
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
print("AUC:", roc_auc_lr)
print("Thresholds:", thresholds_lr)


# Define the IV threshold for feature selection
iv_threshold = 0.2  # Adjust this threshold as needed

def create_bins_dynamic(data, min_percentage=0.05):
    """
    Create bins for continuous data dynamically ensuring each bin has at least min_percentage of the values.
    
    Parameters:
        data (Series or array-like): The continuous data to be binned.
        min_percentage (float): The minimum percentage of values that should be in each bin.
    
    Returns:
        list: List of bin edges.
    """
    # Sort the data
    sorted_data = sorted(data)
    
    # Calculate the number of bins needed to satisfy the minimum percentage constraint
    num_bins = int(1 / min_percentage)
    
    # Calculate the bin size
    bin_size = len(data) // num_bins
    
    # Initialize bins list with the minimum value
    bins = [sorted_data[0]]
    
    # Iterate over the sorted data to determine bin edges
    for i in range(bin_size, len(sorted_data), bin_size):
        # Add bin edge if it satisfies the minimum percentage constraint
        if i < len(sorted_data) and (i / len(sorted_data)) >= min_percentage:
            bins.append(sorted_data[i])
    
    # Add maximum value as the last bin edge
    bins.append(sorted_data[-1])
    # Sort the bin edges
    bins = sorted(bins)
    return bins


# Calculate WOE and IV# Calculate WOE and IV

# Calculate WOE and IV
def calculate_woe_iv(feature, target):
    if feature.dtype == 'O' or np.issubdtype(feature.dtype, np.integer):  # Check if feature is categorical or integer
        event_counts = target.groupby(feature).sum()
        non_event_counts = target.groupby(feature).count() - event_counts
    else:  # If feature is continuous
        # Dynamic bin creation
        bin_edges = create_bins_dynamic(feature)
        feature_bins = pd.cut(feature, bins=bin_edges, include_lowest=True)
        
        # Calculate number of events and non-events for each category
        event_counts = target.groupby(feature_bins).sum()
        non_event_counts = target.groupby(feature_bins).count() - event_counts
    
    # Check if any denominator is zero
    if (event_counts == 0).any() or (non_event_counts == 0).any():
        print("Error: One of the denominators is zero.")
        print(event_counts, non_event_counts)
        return None, None
    
    # Calculate proportions
    event_proportion = event_counts / target.sum()
    non_event_proportion = non_event_counts / (target.count() - target.sum())
    
    # Calculate WOE
    woe = np.log(non_event_proportion / event_proportion)
    
    # Calculate IV
    iv = np.sum((non_event_proportion - event_proportion) * woe)
    
    return woe, iv


# Calculate WOE and IV for each feature
woe_iv_values = {}
for col in df.columns:
    if col != 'Churn':  # Exclude 'Churn' column
        # Your code for binning goes here
        woe, iv = calculate_woe_iv(df[col], df['Churn']) 
        if woe is not None and iv is not None:
            woe_iv_values[col] = (woe, iv)

# Display WOE and IV values
for col, (woe, iv) in woe_iv_values.items():
    print(f"Feature: {col}, IV: {iv}")

# Filter features based on IV threshold
selected_features = [col for col, (_, iv) in woe_iv_values.items() if iv > iv_threshold]
print("Selected features based on IV:",selected_features)
print(f"{feature}: {iv_value}")



# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    df[selected_features],  
    df['Churn'],  # Replace 'stroke' with 'Churn'
    test_size=0.2,  
    random_state=0)  

# Train a logistic regression model using transformed variables
model_woe = LogisticRegression()
model_woe.fit(X_train, y_train)

# Make predictions
y_pred_woe = model_woe.predict(X_test)

# Evaluate the model
print("\nModel with selected features based on IV:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_woe))

def plot_heatmap_with_counts(matrix):
    plt.figure(figsize=(12, 7))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5,
                annot_kws={"color": "black"})
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            plt.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
    plt.title('Confusion Matrix with Annotated Counts')
    plt.show()

# Plot confusion matrix
plot_heatmap_with_counts(confusion_matrix(y_test, y_pred))

# Calculate ROC and AUC for the model
y_test_pred_lr_woe = model_woe.predict_proba(X_test)[:, 1]
fpr_lr_woe, tpr_lr_woe, thresholds_lr_woe = roc_curve(y_test, y_test_pred_lr_woe)
roc_auc_lr_woe = auc(fpr_lr_woe, tpr_lr_woe)

# Plot ROC curve
plt.plot(fpr_lr_woe, tpr_lr_woe, label='Logistic Regression with selected features based on IV (AUC = %0.2f)' % roc_auc_lr_woe)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with selected features based on IV')
plt.legend()
plt.show()

print("AUC with selected features based on IV:", roc_auc_lr_woe)
print("Thresholds with selected features based on IV:", thresholds_lr_woe)