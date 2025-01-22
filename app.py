from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load data from CSV files (2020-2024)
df_2021 = pd.read_csv('data/cutoff_2021.csv')
df_2022 = pd.read_csv('data/cutoff_2022.csv')
df_2023 = pd.read_csv('data/cutoff_2023.csv')
df_2024 = pd.read_csv('data/cutoff_cleaned.csv')

# Drop rows with missing values in key columns
cutoff_columns = ['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca']
df_2021 = df_2021.dropna(subset=cutoff_columns)
df_2022 = df_2022.dropna(subset=cutoff_columns)
df_2023 = df_2023.dropna(subset=cutoff_columns)
df_2024 = df_2024.dropna(subset=cutoff_columns)

# Combine all data into a single DataFrame
df = pd.concat([df_2024, df_2022, df_2021, df_2023], ignore_index=True)

# Create a mapping for categorical variables
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

# Function to predict cutoff for 2025 and provide suggestions
def predict_cutoff(college, department, caste, your_cutoff):
    cutoff_column = get_cutoff_column(caste)
    
    if not cutoff_column:
        return "Invalid caste", [], []
    
    if college not in df['college_name'].unique():
        return "Invalid college", [], []
    
    if department not in df['department'].unique():
        return "Invalid department", [], []

    # Prepare data for model training
    X = df[['college_name_code', 'department_code']]
    y = df[cutoff_column]

    # Drop rows with missing cutoff values
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    # One-hot encoding and pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['college_name_code', 'department_code'])
        ]
    )
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('model', Ridge(alpha=1.0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model_pipeline.fit(X_train, y_train)

    college_code = df.loc[df['college_name'] == college, 'college_name_code'].values[0]
    department_code = df.loc[df['department'] == department, 'department_code'].values[0]

    input_data = pd.DataFrame([[college_code, department_code]], columns=['college_name_code', 'department_code'])
    predicted_cutoff = model_pipeline.predict(input_data)[0]
    predicted_cutoff = max(77, min(predicted_cutoff, 200))

    # Filter data based on cutoff
    df_filtered = df[df[cutoff_column] <= your_cutoff]

    # Get top 5 departments in the same college
    college_suggestions = df_filtered[df_filtered['college_name'] == college]
    college_suggestions = college_suggestions.sort_values(by=cutoff_column, ascending=False)
    college_suggestions = college_suggestions.drop_duplicates(subset=['department'])[
        ['college_name', 'department', cutoff_column]
    ].head(5).to_dict('records')

    # Get top 5 colleges for the same department
    department_suggestions = df_filtered[df_filtered['department'] == department]
    department_suggestions = department_suggestions.sort_values(by=cutoff_column, ascending=False)
    department_suggestions = department_suggestions.drop_duplicates(subset=['college_name'])[
        ['college_name', 'department', cutoff_column]
    ].head(5).to_dict('records')

    return predicted_cutoff, college_suggestions, department_suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    college = request.form['college']
    department = request.form['department']
    caste = request.form['caste']
    your_cutoff = float(request.form['your_cutoff'])

    predicted_cutoff, college_suggestions, department_suggestions = predict_cutoff(
        college, department, caste, your_cutoff
    )

    if isinstance(predicted_cutoff, str):
        return render_template('index.html', prediction=predicted_cutoff)

    eligible = your_cutoff >= predicted_cutoff

    return render_template(
        'index.html',
        prediction=f"{predicted_cutoff:.2f}",
        your_cutoff=your_cutoff,
        eligible=eligible,
        college_suggestions=college_suggestions,
        department_suggestions=department_suggestions,
        selected_college=college,
        selected_department=department
    )

if __name__ == "__main__":
    app.run(debug=True)