HomeBrain Property Price Predictor 🏡

Overview
HomeBrain is a property price predictor application built with Flask, which leverages the RandomForest Regressor to estimate property prices and assist users in discovering properties listed below their market value. This project was undertaken as part of the Berkeley Extension Data BootCamp, with a dedicated team working over two weeks to bring this idea to life.

Application Screenshot
Replace the above link with a screenshot of your application for a visual overview.

Features 🌟
Price Prediction: Utilizes a RandomForest Regressor to predict property prices based on historical data.
Deal Finder: Assists users in finding properties that are potentially undervalued.
Data Visualizations: Interactive graphs and charts that offer insights into property trends.
Performance Metrics: Displays metrics such as MAE, MSE, RMSE, and R^2 to provide transparency into the model's accuracy and performance.

Features

1. **Train Model**: Uses a `RandomForestRegressor` model to predict property prices based on various features. The model is trained on data from a `Sold.csv` file.
2. **Performance Metrics**: Displays metrics like MAE, MSE, RMSE, and R² to show how well the model performs.
3. **Find Deals**: Helps users find property listings that are potential good deals based on the trained model and a user-defined budget.
4. **Analysis**: Visualizes correlations among different property features.
5. **Feature Importances**: Shows which features are most influential in the model's predictions.
6. **Property Scraper**: Gathers data from property listings on websites and saves it to a CSV file.

## Technical Details

### Technologies and Libraries Used:

- Flask: Used for backend and web application.
- Pandas: Data manipulation and analysis.
- Scikit-learn: Data preprocessing and machine learning model.
- Plotly: Data visualization.
- BeautifulSoup: Web scraping.

### Key Components:

1. **Data Preprocessing**:
    - Cleaning functions to handle special characters in price and numerical data.
    - Functions to preprocess and encode data using `OneHotEncoder`.
    
2. **Model Training**:
    - Utilizes `RandomForestRegressor` for the prediction model.
    - Employs `Pipeline` and `ColumnTransformer` for streamlined preprocessing and model fitting.
    
3. **Web Application Endpoints**:
    - `/`: Homepage.
    - `/performance_metrics`: Shows model performance metrics.
    - `/find_deals`: Helps users find potential property deals.
    - `/analysis`: Displays property feature analysis.
    - `/feature_importances`: Visualizes feature importances from the model.
    
4. **Web Scraping**:
    - Uses `requests` and `BeautifulSoup` to scrape property data from web pages.
    - Extracts details like property name, address, price, square footage, number of beds/baths, property type, and year built.
    - Saves scraped data to a CSV file.
      
Dataset 📊
The data used for training the model is derived from a real estate CSV file which includes features like Walk Score, Transit Score, Bike Score, School Scores, Beds, Baths, Square footage, and more.

Team 👥
Binh Do: Team Leader, Backend Developer, Machine Learning Engineer
Beenish: Data Analyst
Nasr: Frontend Backend Analyst
Joy: Data Analyst 
Include brief descriptions or roles if you'd like.

Acknowledgments 🙏
A huge shoutout to our mentors at Berkeley Extension Data BootCamp for their guidance and support throughout this project's duration.

Contributing and Feedback 🤝
Contributions, feedback, and suggestions are welcome! Open an issue or submit a pull request.
