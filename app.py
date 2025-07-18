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

# Combine all data into a single DataFrame
df = pd.concat([df_2024, df_2022, df_2021, df_2023], ignore_index=True)

# Create a mapping for categorical variables
df['college_name_code'] = df['college_name'].astype('category').cat.codes
df['department_code'] = df['department'].astype('category').cat.codes

# List of cutoff columns to process
cutoff_columns = ['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca']

# Fill missing values with 77.5 instead of dropping rows
df_2021[cutoff_columns] = df_2021[cutoff_columns].fillna(77.5)
df_2022[cutoff_columns] = df_2022[cutoff_columns].fillna(77.5)
df_2023[cutoff_columns] = df_2023[cutoff_columns].fillna(77.5)
df_2024[cutoff_columns] = df_2024[cutoff_columns].fillna(77.5)

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

def round_to_nearest_half_or_whole(num):
    """Round number to nearest .00 or .50"""
    base = int(num)
    decimal = num - base
    if decimal < 0.25:
        return float(base)
    elif decimal < 0.75:
        return base + 0.50
    else:
        return float(base + 1)

# Prediction function (unchanged)
def predict_cutoff(college, department, caste, your_cutoff):
    cutoff_column = get_cutoff_column(caste)
    
    if not cutoff_column:
        return "Invalid caste", [], [], [], {}
    
    if college not in df['college_name'].unique():
        return "Invalid college", [], [], [], {}
    
    if department not in df['department'].unique():
        return "Invalid department", [], [], [], {}

    X = df[['college_name_code', 'department_code']]
    y = df[cutoff_column]

    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

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

    rounded_predicted_cutoff = round_to_nearest_half_or_whole(predicted_cutoff)

    df_filtered = df[df[cutoff_column] <= your_cutoff]

    college_suggestions = df_filtered[df_filtered['college_name'] == college]
    college_suggestions = college_suggestions.sort_values(by=cutoff_column, ascending=False)
    college_suggestions = college_suggestions.drop_duplicates(subset=['department'])[
        ['college_name', 'department', cutoff_column]
    ].head(5).to_dict('records')

    department_suggestions = df_filtered[df_filtered['department'] == department]
    department_suggestions = department_suggestions.sort_values(by=cutoff_column, ascending=False)
    department_suggestions = department_suggestions.drop_duplicates(subset=['college_name'])[
        ['college_name', 'department', cutoff_column]
    ].head(5).to_dict('records')
    
    top_colleges = df_filtered.sort_values(by=cutoff_column, ascending=False)
    top_colleges = top_colleges.drop_duplicates(subset=['college_name'])[
        ['college_name', 'department', cutoff_column]
    ].head(5).to_dict('records')
    
    college_avg_cutoff = df[df['college_name'] == college][cutoff_column].mean()
    department_avg_cutoff = df[df['department'] == department][cutoff_column].mean()
    overall_avg_cutoff = df[cutoff_column].mean()
    
    worth_assessment = {
        'college_avg_cutoff': round(college_avg_cutoff, 2),
        'department_avg_cutoff': round(department_avg_cutoff, 2),
        'overall_avg_cutoff': round(overall_avg_cutoff, 2),
        'difference_from_college_avg': round(predicted_cutoff - college_avg_cutoff, 2),
        'difference_from_department_avg': round(predicted_cutoff - department_avg_cutoff, 2),
        'difference_from_overall_avg': round(predicted_cutoff - overall_avg_cutoff, 2),
        'is_worth_it': None
    }

    if your_cutoff > predicted_cutoff + 5:
        worth_assessment['is_worth_it'] = 'high_value'
    elif your_cutoff >= predicted_cutoff:
        worth_assessment['is_worth_it'] = 'good_value'
    elif your_cutoff >= predicted_cutoff - 5:
        worth_assessment['is_worth_it'] = 'fair_value'
    else:
        worth_assessment['is_worth_it'] = 'low_value'

    return predicted_cutoff, college_suggestions, department_suggestions, top_colleges, worth_assessment

@app.route('/')
def home():
    df = pd.read_csv('data/cutoff_cleaned.csv')
    colleges = df['college_name'].unique().tolist()
    college_departments = {
        college: df[df['college_name'] == college]['department'].unique().tolist()
        for college in colleges
    }
    return render_template('index.html', colleges=colleges, college_departments=college_departments)

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('data/cutoff_cleaned.csv')
    colleges = df['college_name'].unique().tolist()
    college_departments = {
        college: df[df['college_name'] == college]['department'].unique().tolist()
        for college in colleges
    }
    college = request.form['college']
    department = request.form['department']
    caste = request.form['caste']
    your_cutoff = float(request.form['your_cutoff'])

    print(college, department, caste, your_cutoff)

    predicted_cutoff, college_suggestions, department_suggestions, top_colleges, worth_assessment = predict_cutoff(
        college, department, caste, your_cutoff
    )

    if isinstance(predicted_cutoff, str):
        return render_template('index.html', prediction=predicted_cutoff)

    eligible = your_cutoff >= predicted_cutoff

    return render_template(
        'index.html',
        prediction=f"{round(predicted_cutoff):.2f}",
        your_cutoff=your_cutoff,
        colleges=colleges,
        college_departments=college_departments,
        eligible=eligible,
        college_suggestions=college_suggestions,
        department_suggestions=department_suggestions,
        top_colleges=top_colleges,
        worth_assessment=worth_assessment,
        selected_college=college,
        selected_department=department
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
