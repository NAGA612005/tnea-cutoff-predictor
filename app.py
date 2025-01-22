from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

# Load data from CSV files (2020-2024)
df_2021 = pd.read_csv('data/cutoff_2021.csv')
df_2022 = pd.read_csv('data/cutoff_2022.csv')
df_2023 = pd.read_csv('data/cutoff_2023.csv')
df_2024 = pd.read_csv('data/cutoff_cleaned.csv')

# Drop rows with missing values in key columns
df_2021 = df_2021.dropna(subset=['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca'])
df_2022 = df_2022.dropna(subset=['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca'])
df_2023 = df_2023.dropna(subset=['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca'])
df_2024 = df_2024.dropna(subset=['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca'])

# Combine all data into a single DataFrame
df = pd.concat([df_2024, df_2022, df_2021, df_2023], ignore_index=True)

# Create a mapping for categorical variables (college_name, department)
df['college_name_code'] = df['college_name'].astype('category').cat.codes
df['department_code'] = df['department'].astype('category').cat.codes

# Function to get the cutoff column based on caste
def get_cutoff_column(caste):
    caste_cutoff_map = {
        'oc': 'cutoff_oc',
        'bc': 'cutoff_bc',
        'mbc': 'cutoff_mbc',
        'bcm': 'cutoff_bcm',
        'sc': 'cutoff_sc',
        'st': 'cutoff_st',
        'sca': 'cutoff_sca'
    }
    return caste_cutoff_map.get(caste.lower())

# Function to predict cutoff for 2025
def predict_cutoff(college, department, caste):
    cutoff_column = get_cutoff_column(caste)

    if not cutoff_column:
        return "Invalid caste"

    # Check if college and department are valid
    if college not in df['college_name'].unique():
        return "Invalid college"

    if department not in df['department'].unique():
        return "Invalid department"

    # Prepare data for model training
    X = df[['college_name_code', 'department_code']]
    y = df[cutoff_column]

    # Drop rows with missing cutoff values
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    # One-hot encoding for categorical variables (college_name and department)
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['college_name_code', 'department_code'])
        ])

    # Create a pipeline that includes the preprocessor, standard scaler, and Ridge regression model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),  # Scaling features
        ('model', Ridge(alpha=1.0))    # Ridge Regression
    ])

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Get college and department codes
    college_code = df.loc[df['college_name'] == college, 'college_name_code'].values[0]
    department_code = df.loc[df['department'] == department, 'department_code'].values[0]

    # Predict for 2025 using the pipeline
    input_data = pd.DataFrame([[college_code, department_code]], columns=['college_name_code', 'department_code'])
    predicted_cutoff = model_pipeline.predict(input_data)[0]

    # Post-process prediction to ensure it does not exceed 200 and is within a reasonable range
    predicted_cutoff = max(77, min(predicted_cutoff, 200))

    return predicted_cutoff  # Return the predicted value

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Result page after prediction
@app.route('/predict', methods=['POST'])
def predict():
    college = request.form['college']
    department = request.form['department']
    caste = request.form['caste']
    your_cutoff = float(request.form['your_cutoff'])  # Convert to float for comparison

    # Call the prediction function
    predicted_cutoff = predict_cutoff(college, department, caste)
    
    if isinstance(predicted_cutoff, str):  # If it's an error message
        return render_template('index.html', prediction=predicted_cutoff, your_cutoff=your_cutoff)

    # Check eligibility
    eligible = your_cutoff >= predicted_cutoff  # Compare numerical values

    return render_template('index.html', prediction=f"{predicted_cutoff:.2f}", your_cutoff=your_cutoff, eligible=eligible)

if __name__ == "__main__":
    app.run(debug=True)
