from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Global variables to store the model and transformer
model = None
preprocessor = None
metrics = None
Training_data = None
original_data = None

drop_cols = ['URL', 'Address', 'Property Type', 'Full Address', 'Price']

def clean_hoa(hoa_str):
    return float(hoa_str.replace('$', '').replace(',', '').strip())


def clean_price(price_str):
    return float(price_str.replace('$', '').replace(',', '').strip()) if isinstance(price_str, str) else price_str


def clean_numerical(num_str):
    return float(num_str.replace(',', '').strip()) if isinstance(num_str, str) else num_str


def preprocess_data(train_data, new_data):
    train_data = train_data.reset_index(drop=True)
    new_data = new_data.reset_index(drop=True)
    merged_data = pd.concat([train_data, new_data], sort=False).reset_index(drop=True)
    merged_data_encoded = pd.get_dummies(merged_data)
    train_data_encoded = merged_data_encoded.loc[:len(train_data) - 1]
    new_data_encoded = merged_data_encoded.loc[len(train_data):].reset_index(drop=True)
    return train_data_encoded, new_data_encoded


def train_model():
    global model, preprocessor, metrics, Training_data, original_data
    try:
        data = pd.read_csv('static/csv/Sold.csv')
        data['Price'] = data['Price'].apply(clean_price)
        data['HOA Dues'] = data['HOA Dues'].apply(clean_hoa)
        numerical_cols = ['Walk Score (out of 100)', 'Transit Score (out of 100)', 'Bike Score (out of 100)',
            'Elementary School Score (out of 10)', 'Middle School Score (out of 10)',
            'High School Score (out of 10)', 'Price', 'Beds', 'Baths', 'Sq Ft',
            'Year Built', 'HOA Dues', 'Lot Size', 'Garage Spaces', 'Zipcode'
        ]
        for col in numerical_cols:
            data[col] = data[col].apply(clean_numerical)
        categorical_cols = [
            # ... rest of your categorical columns ...
        ]
        X = data[numerical_cols + categorical_cols]
        y = data['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        Training_data = X_train
        original_data = data
        # Preprocessing for numerical and categorical data
        numerical_transformer = SimpleImputer(strategy='mean')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

         # Define the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Bundle preprocessing and modeling code in a pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])

        # Preprocessing of training data, train model
        clf.fit(X_train, y_train)

        # Preprocessing of validation data, get predictions
        preds = clf.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Error in train_model: {e}")

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/performance_metrics')
def performance_metrics():
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        return render_template('performance_metrics.html', mae=metrics["MAE"], mse=metrics["MSE"], rmse=metrics["RMSE"], r2=metrics["R2"])
    except FileNotFoundError:
        return "Metrics not available. Please train the model first."

@app.route('/find_deals', methods=['GET', 'POST'])
def find_deals():
    global Training_data, original_data, model
    if model is None or preprocessor is None:
        return "Model or preprocessor not initialized. Please train the model first."

    if request.method == 'POST':
        budget = float(request.form['budget'])

        new_data = pd.read_csv('static/csv/Listing.csv')
        new_data_prep = new_data.drop(drop_cols, axis=1)

        new_data['Price'] = pd.to_numeric(new_data['Price'].str.replace('[\$\,]', ''), errors='coerce')

        for column in new_data_prep.select_dtypes(include=['object']).columns:
            if column in original_data.select_dtypes(include=['category']).columns:
                new_data_prep[column] = new_data_prep[column].astype('category')
                new_data_prep[column] = new_data_prep[column].cat.set_categories(
                    original_data[column].cat.categories
                )

        new_data_encoded = pd.get_dummies(new_data_prep)
        
        # Replace this line with the modified line below
        # new_data_encoded, _ = new_data_encoded.align(training_data, axis=1, fill_value=0)
        new_data_encoded, _ = new_data_encoded.align(Training_data, axis=1, fill_value=0)  # Modified line

        # Ensure the order of columns matches the training data
        new_data_encoded = new_data_encoded[Training_data.columns]

        new_data_imputed = preprocessor.transform(new_data_encoded)


        predicted_prices = model.predict(new_data_imputed)
        new_data['Predicted Price'] = predicted_prices
        new_data['Price Difference'] = new_data['Predicted Price'] - new_data['Price']


        good_deals = new_data[new_data['Price Difference'] > 0]
        affordable_good_deals = good_deals[good_deals['Price'] <= budget]
        affordable_good_deals = affordable_good_deals[['URL', 'Address', 'Price', 'Beds', 'Baths', 'Sq Ft', 'Predicted Price', 'Price Difference']]
        affordable_good_deals = affordable_good_deals.sort_values(by='Price Difference', ascending=False)

        format_price = lambda x: '${:,.0f}'.format(x)
        affordable_good_deals['Price'] = affordable_good_deals['Price'].apply(format_price)
        affordable_good_deals['Predicted Price'] = affordable_good_deals['Predicted Price'].apply(format_price)
        affordable_good_deals['Price Difference'] = affordable_good_deals['Price Difference'].apply(format_price)

        affordable_good_deals['URL'] = affordable_good_deals['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        deals_html_table = affordable_good_deals.to_html(classes='data', header="true", index=False, escape=False)

        return render_template('results.html', deals_table=deals_html_table)

    return render_template('find_deals.html')

@app.route('/analysis')
def analysis():
    heatmap_image_path = 'static/images/heatmap.png'
    scatterplot_image_path = 'static/images/scatterplot.png'
    return render_template('analysis.html', heatmap_image=heatmap_image_path, scatterplot_image=scatterplot_image_path)

def get_feature_names(column_transformer):
    feature_names = []
    for transformer_info in column_transformer.transformers_:
        transformer_name, transformer, columns = transformer_info
        if isinstance(transformer, OneHotEncoder):
            transformed_features = transformer.get_feature_names_out(columns)
            feature_names.extend(transformed_features)
        elif transformer == 'passthrough':
            feature_names.extend(columns)
        else:
            feature_names.extend(columns)
    return feature_names

@app.route('/feature_importances')
def feature_importances():
    global model, preprocessor
    # Ensure the model and preprocessor are initialized
    if model is None or preprocessor is None:
        return "Model or preprocessor not initialized. Please train the model first."

    features = get_feature_names(preprocessor)
    importances = model.feature_importances_

    if len(importances) != len(features):
        return f"Mismatched feature importances and feature names: {len(importances)} importances, {len(features)} features"
    
    
    importances_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(14)
    
    fig = px.bar(importances_df, x='Importance', y='Feature', orientation='h', title='Feature Importances')
    graph_html = pio.to_html(fig, full_html=False)
    
    return render_template('feature_importances.html', graph=graph_html)

if __name__ == '__main__':
    train_model()
    app.run(debug=True)