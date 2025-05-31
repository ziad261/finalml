Smart Home Energy Prediction - Project Documentation
Project Overview
This project aims to build a predictive model for estimating energy consumption in smart homes. Smart homes typically consist of various appliances whose usage varies by season, weather conditions, and human activity. The goal of this project is to provide accurate predictions using machine learning techniques, thus enabling better energy management, cost-saving strategies, and sustainability.
The system is built to be modular, reusable, and scalable, integrating preprocessing, feature engineering, model training with hyperparameter optimization, evaluation using standard metrics, and deployment using a web interface. The project also provides an interactive API for real-time predictions.
Pipeline Workflow
The pipeline consists of the following structured components:
1. preprocessing.py
This module is responsible for loading, cleaning, and scaling the dataset to ensure it is ready for machine learning workflows. It includes two main functions:
🔹 load_and_clean_data(file_path)
•	Reads the CSV file and ensures that the 'Date' column is parsed as a datetime object.
•	Handles missing values:
o	Replaces missing values in 'Appliance Type' with the most frequent category (mode).
o	Fills missing values in 'Energy_Consumption_(kWh)' using the median, which is robust to outliers.
•	Removes any duplicate rows to ensure clean, non-redundant data is used for modeling.
🔹 scale_features(df)
•	Applies standardization using StandardScaler from Scikit-learn.
•	Specifically scales the numerical features:
o	'Energy_Consumption_(kWh)'
o	'Outdoor Temperature (°C)'
•	Standardizing ensures that features have a mean of 0 and standard deviation of 1, improving the convergence speed and stability of many machine learning algorithms.
Together, this module ensures that all raw input data is clean, consistent, and numerically scaled before entering the model training pipeline.
2. feature_engineering.py
This module performs both temporal feature extraction and categorical encoding, preparing the dataset for machine learning model training. It consists of three main functions:
🔹 add_time_features(df)
•	Converts the 'Date' and 'Time' columns into datetime format.
•	Extracts components such as:
o	Year, Month, Day
o	Day of Week
o	Week Number
o	Hour and Minute
•	Additionally, it creates cyclical time features using sine and cosine transformations of the hour:
o	hour_sin and hour_cos
o	This helps the model understand the periodic nature of time-based events (e.g., hourly appliance usage patterns).
🔹 encode_features(df)
•	Encodes categorical variables using Ordinal Encoding, ensuring they are in numerical format.
•	The columns encoded include:
o	'Appliance Type'
o	'Season'
o	'energy_level'
🔹 engineer_features(df)
•	Integrates the above two steps into a complete feature engineering pipeline.
•	Introduces a new feature 'energy_level' based on quantiles of 'Energy_Consumption_(kWh)':
o	Labeled as 'Low', 'Medium', and 'High' based on energy consumption distribution.
•	Separates the features X (input) and target y (Energy_Consumption_(kWh)), and returns them alongside the fitted OrdinalEncoder.
Together, this module transforms raw data into a fully prepared feature matrix and target vector, enabling robust and insightful training of ML models.
3. train_model.py
This module orchestrates the training phase of the machine learning pipeline, including model fitting, hyperparameter optimization, and saving key components for later inference.
🔹 train_and_tune_model(df, scaler=None, encoder=None)
•	Input Separation:
The function separates the target variable 'Energy_Consumption_(kWh)' from the input features, excluding date/time columns if present.
•	Train-Test Split:
Data is split into training and testing subsets using an 80-20 ratio, ensuring robust evaluation of generalization.
•	Hyperparameter Optimization:
Utilizes GridSearchCV on a RandomForestRegressor to explore combinations of:
o	Number of estimators (n_estimators)
o	Tree depth (max_depth)
o	Minimum samples for splitting (min_samples_split)
GridSearch is performed using 3-fold cross-validation, optimizing for negative RMSE to find the best model parameters.
•	Model Selection:
The best-performing model configuration is selected based on the lowest RMSE, and the resulting RandomForestRegressor is retained for deployment.
•	Model Saving:
The final model, the names of the input features used in training, and optionally the scaler and ordinal_encoder are saved in the models/ directory using joblib.
This approach ensures a production-ready model is trained efficiently and stored with all necessary preprocessing artifacts for future use in deployment or batch inference.

4. model_evaluation.py
This module is responsible for evaluating the performance of the trained machine learning model using a test dataset that the model has not seen during training. This ensures an unbiased estimate of how well the model generalizes to new data.
The evaluation is performed using the following key metrics:
•	Root Mean Square Error (RMSE):
This measures the square root of the average squared differences between predicted and actual values. RMSE penalizes large errors more than small ones, making it suitable for highlighting significant prediction deviations. A lower RMSE indicates a better fit.
•	Mean Absolute Error (MAE):
This metric calculates the average absolute difference between actual and predicted values. MAE is straightforward to interpret and is less sensitive to outliers than RMSE.
•	R² Score (Coefficient of Determination):
The R² score explains the proportion of variance in the target variable that is predictable from the input features. A value close to 1.0 indicates that the model explains most of the variability, while a value near 0 indicates poor predictive power.
Together, these metrics provide a comprehensive understanding of:
•	Accuracy (how close predictions are to actual values),
•	Robustness (how well the model handles unseen data), and
•	Consistency (how much variability is explained).
5. deploy/app.py
This module provides a Flask-based REST API for serving the trained machine learning model. It enables real-time prediction of energy consumption using HTTP POST requests and integrates preprocessing steps for seamless inference.
🔹 Core Features:
•	Web Interface (/):
The root endpoint renders a basic HTML template (index.html) that can optionally be extended into a user interface.
•	API Endpoint (/predict):
Accepts POST requests with JSON input containing feature values required by the trained model.
🔹 Processing Steps:
1.	Input Handling:
Receives JSON data and converts it into a Pandas DataFrame for compatibility with Scikit-learn models.
2.	Categorical Encoding:
Uses a pre-trained OrdinalEncoder to encode categorical inputs such as:
o	'Appliance Type'
o	'Season'
o	'energy_level'
3.	Numerical Scaling:
Applies a saved StandardScaler to normalize numeric values like 'Outdoor Temperature (°C)'.
4.	Feature Alignment:
Loads the feature_names.pkl file to ensure the input features match the exact order and structure used during training.
5.	Prediction:
Passes the processed input to the Random Forest model and returns the predicted energy consumption value in kWh.
🛠️ Setup Instructions
To run the Smart Home Energy Prediction pipeline and API on your local machine, follow these steps:
1. Environment Setup
Ensure you have Python 3.7 or higher installed. Then install all required libraries using the command below:
pip install -r requirements.txt
The requirements file typically includes: pandas, numpy, scikit-learn, flask, and joblib.
If the file is missing or incomplete, you can manually install with:
pip install flask pandas scikit-learn joblib
2. Project Structure
Your folder should be structured as follows:
final_ml/
├── ml_pipeline/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── model_evaluation.py
│   ├── deploy/
│   │   ├── app.py
│   │   └── templates/
│   │       └── index.html
├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── ordinal_encoder.pkl
│   ├── feature_names.pkl
├── smart_home_energy_modified.csv
├── main.py
3. Running the Full Pipeline
You can run each module individually using:
python -m ml_pipeline.preprocessing
python -m ml_pipeline.feature_engineering
python -m ml_pipeline.train_model
python -m ml_pipeline.model_evaluation
Alternatively, create a unified script (e.g., main.py) to automate the pipeline end-to-end.
4. Running the Flask API
To launch the API for real-time predictions:
python ml_pipeline/deploy/app.py
Access the web interface at http://localhost:5000 and send POST requests to /predict with a JSON body to receive energy predictions.
📩 Making Predictions via API
Once the Flask API is running, you can make predictions by sending an HTTP POST request to the `/predict` endpoint. This allows real-time interaction with the trained model using structured JSON input.
1. Input Format
The API expects a JSON object that includes all the input features used during training. An example input looks like this:
{
  "Appliance Type": "Microwave",
  "Season": "Winter",
  "energy_level": "Medium",
  "Outdoor Temperature (°C)": 20
}
Ensure that all categorical values match those used in the training dataset. Missing or incorrect keys may cause the prediction to fail.
Output Artifacts
Upon successful execution of the training and deployment pipeline, the following output artifacts are generated and stored in the `models/` directory. These files are essential for prediction and deployment:
•	- `rf_model.pkl`
  This is the trained Random Forest Regressor model used for making energy consumption predictions.
•	- `scaler.pkl`
  This file contains the fitted StandardScaler, used to normalize numerical input features such as 'Outdoor Temperature (°C)'.
•	- `ordinal_encoder.pkl`
  This file contains the trained OrdinalEncoder used for converting categorical values like 'Appliance Type', 'Season', and 'energy_level' into numeric format.
•	- `feature_names.pkl`
  Stores the ordered list of feature names used during model training to ensure consistent alignment during inference.
Summary
This project presents a complete machine learning pipeline for predicting smart home energy consumption. It covers essential stages including data preprocessing, temporal and categorical feature engineering, model training with hyperparameter tuning, evaluation using multiple metrics, and deployment through a Flask API.
The system is designed with modularity and scalability in mind. Key model artifacts like the trained Random Forest, scaler, and encoder are preserved for seamless deployment and inference. By combining engineering best practices and robust machine learning techniques, the project offers a reliable, real-time energy prediction solution suitable for integration into smart home systems and IoT platforms.
This approach can be extended to support additional features, advanced model ensembling, and frontend interfaces for user interaction, making it a strong foundation for future smart energy applications.

