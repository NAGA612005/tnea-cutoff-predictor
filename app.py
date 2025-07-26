from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

CUTOFF_COLUMNS = ['cutoff_oc', 'cutoff_bc', 'cutoff_mbc', 'cutoff_bcm', 'cutoff_sc', 'cutoff_st', 'cutoff_sca']
CSV_FILES = ['data/cutoff_cleaned.csv', 'data/cutoff_2022.csv', 'data/cutoff_2021.csv', 'data/cutoff_2023.csv']

def load_and_prepare_data():
    dfs = []
    for file in CSV_FILES:
        df = pd.read_csv(file)
        df[CUTOFF_COLUMNS] = df[CUTOFF_COLUMNS].fillna(77.5)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['college_name_code'] = combined_df['college_name'].astype('category').cat.codes
    combined_df['department_code'] = combined_df['department'].astype('category').cat.codes
    return combined_df

df = load_and_prepare_data()

unique_colleges = df['college_name'].unique().tolist()
college_departments = {
    college: df[df['college_name'] == college]['department'].unique().tolist()
    for college in unique_colleges
}

def get_cutoff_column(caste):
    return {
        'oc': 'cutoff_oc', 'bc': 'cutoff_bc', 'mbc': 'cutoff_mbc',
        'bcm': 'cutoff_bcm', 'sc': 'cutoff_sc', 'st': 'cutoff_st', 'sca': 'cutoff_sca'
    }.get(caste.lower())

def round_to_nearest_half_or_whole(num):
    base = int(num)
    decimal = num - base
    if decimal < 0.25:
        return float(base)
    elif decimal < 0.75:
        return base + 0.5
    else:
        return float(base + 1)

# Prediction Logic
def predict_cutoff(college, department, caste, your_cutoff):
    cutoff_column = get_cutoff_column(caste)
    if not cutoff_column:
        return "Invalid caste", [], [], [], {}

    if college not in df['college_name'].unique():
        return "Invalid college", [], [], [], {}

    if department not in df['department'].unique():
        return "Invalid department", [], [], [], {}

    filtered_df = df[df[cutoff_column].notna()]
    X = filtered_df[['college_name_code', 'department_code']]
    y = filtered_df[cutoff_column]

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('onehot', OneHotEncoder(), ['college_name_code', 'department_code'])
        ])),
        ('scaler', StandardScaler(with_mean=False)),
        ('model', Ridge(alpha=3.0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    pipeline.fit(X_train, y_train)

    college_code = df[df['college_name'] == college]['college_name_code'].values[0]
    department_code = df[df['department'] == department]['department_code'].values[0]
    input_data = pd.DataFrame([[college_code, department_code]], columns=['college_name_code', 'department_code'])

    predicted = pipeline.predict(input_data)[0]
    predicted = max(77, min(predicted, 200))
    rounded_prediction = round_to_nearest_half_or_whole(predicted)

    df_filtered = df[df[cutoff_column] <= your_cutoff]

    college_suggestions = df_filtered[df_filtered['college_name'] == college].sort_values(by=cutoff_column, ascending=False)
    college_suggestions = college_suggestions.drop_duplicates('department')[['college_name', 'department', cutoff_column]].head(5).to_dict('records')

    department_suggestions = df_filtered[df_filtered['department'] == department].sort_values(by=cutoff_column, ascending=False)
    department_suggestions = department_suggestions.drop_duplicates('college_name')[['college_name', 'department', cutoff_column]].head(5).to_dict('records')

    top_colleges = df_filtered.sort_values(by=cutoff_column, ascending=False)
    top_colleges = top_colleges.drop_duplicates('college_name')[['college_name', 'department', cutoff_column]].head(5).to_dict('records')

    worth = {
        'college_avg_cutoff': round(df[df['college_name'] == college][cutoff_column].mean(), 2),
        'department_avg_cutoff': round(df[df['department'] == department][cutoff_column].mean(), 2),
        'overall_avg_cutoff': round(df[cutoff_column].mean(), 2)
    }
    worth.update({
        'difference_from_college_avg': round(predicted - worth['college_avg_cutoff'], 2),
        'difference_from_department_avg': round(predicted - worth['department_avg_cutoff'], 2),
        'difference_from_overall_avg': round(predicted - worth['overall_avg_cutoff'], 2)
    })

    if your_cutoff > predicted + 5:
        worth['is_worth_it'] = 'high_value'
    elif your_cutoff >= predicted:
        worth['is_worth_it'] = 'good_value'
    elif your_cutoff >= predicted - 5:
        worth['is_worth_it'] = 'fair_value'
    else:
        worth['is_worth_it'] = 'low_value'

    return rounded_prediction, college_suggestions, department_suggestions, top_colleges, worth

@app.route('/')
def home():
    return render_template('index.html', colleges=unique_colleges, college_departments=college_departments)

@app.route('/predict', methods=['POST'])
def predict():
    college = request.form['college']
    department = request.form['department']
    caste = request.form['caste']
    your_cutoff = float(request.form['your_cutoff'])

    predicted_cutoff, college_suggestions, department_suggestions, top_colleges, worth_assessment = predict_cutoff(
        college, department, caste, your_cutoff
    )

    if isinstance(predicted_cutoff, str):
        return render_template('index.html', prediction=predicted_cutoff, colleges=unique_colleges, college_departments=college_departments)

    eligible = your_cutoff >= predicted_cutoff

    return render_template(
        'index.html',
        prediction=f"{predicted_cutoff:.2f}",
        your_cutoff=your_cutoff,
        colleges=unique_colleges,
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
