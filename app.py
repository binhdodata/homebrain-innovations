from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Global variables to store the model and imputer
model = None
imputer = None
training_data = None
original_data = None  # Define original_data here

def train_model():
    global model, imputer, training_data, original_data
    data = pd.read_csv('static/csv/Main Data.csv')

    # Preprocessing steps
    drop_cols = ['URL', 'Elementary School Name', 'Middle School Name', 'High School Name', 
                 'Address', 'Property Type', 'Style', 'Floor Type', 'Heat Type', 'Cool Type',
                 'Noise Level', 'Flood Factor', 'Fire Factor', 'Heat Factor', 'Wind Factor',
                 'Price']
    X = data.drop(drop_cols, axis=1)
    y = data['Price']
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = X[column].astype('category')
    original_data = X.copy()  # Save a copy of the original data before encoding
    X_encoded = pd.get_dummies(X)
    training_data = X_encoded  # Save the training data for later use

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

drop_cols = ['URL', 'Elementary School Name', 'Middle School Name', 'High School Name', 
             'Address', 'Property Type', 'Style', 'Floor Type', 'Heat Type', 'Cool Type',
             'Noise Level', 'Flood Factor', 'Fire Factor', 'Heat Factor', 'Wind Factor',
             'Price']

@app.route('/find_deals', methods=['GET', 'POST'])

def find_deals():
    global training_data, original_data
    
    if request.method == 'POST':
        budget = float(request.form['budget'])

        new_data = pd.read_csv('static/csv/New Data.csv')
        new_data_prep = new_data.drop(drop_cols, axis=1)  # drop_cols is now accessible
        
        # Identify and remove unseen categories
        for column in new_data_prep.select_dtypes(include=['object']).columns:
            if column in original_data.select_dtypes(include=['category']).columns:
                new_data_prep[column] = new_data_prep[column].astype('category')
                new_data_prep[column] = new_data_prep[column].cat.set_categories(
                    original_data[column].cat.categories
                )
        
        new_data_encoded = pd.get_dummies(new_data_prep)
        new_data_encoded, _ = new_data_encoded.align(training_data, axis=1, fill_value=0)
        new_data_imputed = imputer.transform(new_data_encoded.values)  # .values converts DataFrame to NumPy array
        
        predicted_prices = model.predict(new_data_imputed)
        new_data['Predicted Price'] = predicted_prices
        new_data['Price Difference'] = new_data['Price'] - new_data['Predicted Price']
        good_deals = new_data[new_data['Price Difference'] < 0]
        affordable_good_deals = good_deals[good_deals['Price'] <= budget]
        

        



        deals_html_table = affordable_good_deals.to_html(classes='data', header="true", index=False)
        return render_template('results.html', deals_table=deals_html_table)
    
    return render_template('find_deals.html')

if __name__ == '__main__':
    train_model()
    app.run(debug=True)
