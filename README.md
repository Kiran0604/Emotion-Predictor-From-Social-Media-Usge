The **Emotion Predictor From Social Media Usage** project analyzes social media usage data to predict the dominant emotion of a user. It incorporates exploratory data analysis (EDA), data visualizations, and multiple machine learning models to build a robust prediction tool. The project also includes a simple web application to interact with these predictive models.
**Features**
Exploratory Data Analysis: A comprehensive Jupyter Notebook (EDA And Model Building.ipynb) details the EDA process, data cleaning, and visualization of key relationships (e.g., usage time, likes received, platform usage).
Machine Learning Models: Multiple pipelines are provided, including:
Decision Tree (dt_pipeline.pkl)
K-Nearest Neighbors (knn_pipeline.pkl)
Logistic Regression (lr_pipeline.pkl)
Random Forest (rf_pipeline.pkl)
Stacking Model (stacking_model.pkl)
Support Vector Machine (svm_pipeline.pkl)
Visualizations: The repository includes several images illustrating:
Daily usage time versus dominant emotion
Dominant emotion distribution by gender and platform
Likes received per day in relation to emotion
Platform usage frequency
Web Application: A simple web interface (app.py) lets users input social media usage parameters and receive a predicted dominant emotion.
