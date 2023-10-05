# HomeBrain: Property Price Predictor 🏡

HomeBrain is a state-of-the-art property price predictor built on Flask, harnessing the power of RandomForest Regressor. Developed as part of the Berkeley Extension Data BootCamp, our dedicated team put in rigorous effort over two weeks to realize this innovative concept.


## Key Features 🌟

- **Price Prediction**: Employs `RandomForestRegressor` to make property price predictions, trained on the `Sold.csv` dataset.
  
- **Deal Finder**: Aids users in spotting potentially undervalued properties.
  
- **Data Visualizations**: Dynamic graphs and charts showcasing property market trends.
  
- **Performance Metrics**: Transparency into the model's effectiveness with metrics such as MAE, MSE, RMSE, and R^2.
  
- **Feature Insights**: Understand which property features significantly influence predictions.
  
- **Property Scraper**: Efficiently pulls property data from websites, storing them in a CSV format.

## Technical Aspects 🛠

### Utilized Technologies:

- **Flask**: Powers the backend and the web application framework.
  
- **Pandas**: Facilitates data analysis and manipulation.
  
- **Scikit-learn**: Takes care of data preprocessing and the machine learning model.
  
- **Plotly**: For vivid data visualization.
  
- **BeautifulSoup**: Enables web scraping functionalities.

### Core Components:

1. **Data Preprocessing**:
    - Special character handling in price and other numerical fields.
    - Streamlined preprocessing and encoding with `OneHotEncoder`.

2. **Model Mechanics**:
    - The heart is the `RandomForestRegressor`.
    - Seamless integration of `Pipeline` and `ColumnTransformer` for preprocessing and model fitting.

3. **Web Endpoints**:
    - `/`: The landing page.
    - `/performance_metrics`: A peek into model performance.
    - `/find_deals`: Spot exciting property deals.
    - `/analysis`: Deep dive into property feature interrelations.
    - `/feature_importances`: A visual treat of feature significance.

4. **Web Data Collection**:
    - Leveraging `requests` and `BeautifulSoup` for web data extraction.
    - Details such as property name, address, price, square footage, and more are extracted and stored.

## Dataset Insights 📊

Training is based on a comprehensive real estate dataset comprising features such as Walk Score, Transit Score, Bike Score, School Scores, and typical property characteristics like Beds, Baths, and Square footage.

## Meet the Team 👥

- **Binh Do**: The visionary Team Leader, Backend Developer, and Machine Learning Engineer.
  
- **Beenish**: The insightful Data Analyst.
  
- **Nasr**: Our Frontend and Backend Analyst maestro.
  
- **Joy**: Another brilliant Data Analyst.

## Acknowledgments 🙏

Boundless gratitude to our mentors at Berkeley Extension Data BootCamp. Their unwavering guidance ensured this project's success.

## Contributions & Feedback 🤝

We value your thoughts! Feel free to raise issues or pitch in with a pull request.
   
