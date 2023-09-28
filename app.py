from flask import Flask, request, render_template, redirect, url_for
import model  # Assuming model.py contains necessary functions and model setup

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/find_deals', methods=['GET', 'POST'])
def find_deals():
    deals = None  # Initialize deals to None for GET requests
    if request.method == 'POST':
        budget = request.form.get('budget')  # Use .get to avoid KeyError
        profit = request.form.get('profit')  # Use .get to avoid KeyError
        if budget and profit:  # Check if budget and profit are provided
            try:
                # Assumes find_deals function is defined in model.py and requires budget and profit as float
                deals = model.find_deals(float(budget), float(profit))  
            except ValueError:  # Catch ValueError in case of invalid float conversion
                pass  # You may want to add error handling/logging here
    return render_template('find_deals.html', deals=deals)

if __name__ == '__main__':
    app.run(debug=True)
