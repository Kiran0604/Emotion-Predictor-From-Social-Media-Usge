The **Emotion Predictor App** leverages machine learning models to analyze social media usage data and predict the dominant emotion of a user. It incorporates exploratory data analysis (EDA), data visualizations, and multiple pre-trained machine learning models to build a robust prediction tool. The app also features a simple web application for interacting with these predictive models and a weighted voting system that combines predictions from different models to enhance accuracy.   

**Key Features**  
**Exploratory Data Analysis:** A comprehensive Jupyter Notebook (EDA And Model Building.ipynb) details the EDA process, data cleaning, and visualization of key relationships (e.g., usage time, likes received, platform usage).
**Multiple Models:** Includes Random Forest, KNN, Decision Tree, Logistic Regression, and SVM.   
**Weighted Voting:** Combines model predictions based on their accuracy to provide the most reliable outcome.   
**Custom Inputs:** Enter personalized data such as age, gender, platform, and daily usage statistics to get tailored predictions.   
**Prediction History:** View past predictions along with the input data for further insights.   
**Visual Insights:** Understand the relationship between emotions and different user behaviors (e.g., daily usage, likes received).   

**Models**    
The project provides several pre-trained machine learning pipelines that can be used individually or in combination (via the stacking model) for improved prediction performance:   
Decision Tree Pipeline: dt_pipeline.pkl
K-Nearest Neighbors Pipeline: knn_pipeline.pkl
Logistic Regression Pipeline: lr_pipeline.pkl
Random Forest Pipeline: rf_pipeline.pkl
Stacking Model: stacking_model.pkl
Support Vector Machine Pipeline: svm_pipeline.pkl

**Web Application:** A simple web interface (app.py) lets users input social media usage parameters and receive a predicted dominant emotion.
