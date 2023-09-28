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
    if request.method == 'POST':
        budget = request.form['budget']
        profit = request.form['profit']
        deals = model.find_deals(budget, profit)  # Assumes find_deals function is defined in model.py
        return render_template('find_deals.html', deals=deals)
    return render_template('find_deals.html', deals=None)

if __name__ == '__main__':
    app.run(debug=True)
