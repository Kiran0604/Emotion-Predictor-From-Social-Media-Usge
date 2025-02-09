import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import os

# Load pre-trained models
rf_pipeline = joblib.load('rf_pipeline.pkl')
knn_pipeline = joblib.load('knn_pipeline.pkl')
dt_pipeline = joblib.load('dt_pipeline.pkl')
lr_pipeline = joblib.load('lr_pipeline.pkl')
svm_pipeline = joblib.load('svm_pipeline.pkl')
stacking_model = joblib.load('stacking_model.pkl')

# Pre-trained models
models = {
    'RandomForest': rf_pipeline,
    'KNN': knn_pipeline,
    'DecisionTree': dt_pipeline,
    'LogisticRegression': lr_pipeline,
    'SVM': svm_pipeline,
    'Stacking': stacking_model  
}

# Weights for each model based on accuracy
model_weights = {
    'RandomForest': 5,
    'KNN': 4,
    'DecisionTree': 3,
    'SVM': 3,
    'LogisticRegression': 2,
    'Stacking': 6
}

# Create SQLite database for storing predictions
conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    input_data TEXT,
    predicted_emotion TEXT
)
""")
conn.commit()

# Streamlit UI Setup
st.set_page_config(page_title="Emotion Predictor", layout="centered")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Introduction", "Emotion Predictor", "EDA", "Prediction History","Back to Home"])
if page == "Back to Home":
    st.title("🎉 Thank You for Exploring Emotion Predictor!")
    st.write("""
        We truly appreciate your time in exploring this AI-powered Emotion Predictor.  
        Your engagement helps us refine and enhance our insights for a better experience.  
    """)

    # Divider
    st.markdown("---")

    # Explore More Features Section
    st.header("✨ Explore More Features")
    st.write("""
    - 🧠 **Emotion Prediction**: AI-driven insights into emotions based on user behavior.  
    - 📊 **Exploratory Data Analysis (EDA)**: Interactive visualizations of emotion trends.  
    - 🔍 **Prediction History**: Access and analyze your past emotion predictions.  
    - 🏆 **Multi-Model Voting**: Predictions using multiple AI models with accuracy-based weighting.  
    """)

    # Encouraging further exploration
    st.write("🔎 *Explore different sections from the sidebar before heading back home!*")

    # Manual Redirection Button
    if st.button("🏠 Go to Home"):
        st.markdown('<meta http-equiv="refresh" content="0;URL=http://127.0.0.1:5000/">', unsafe_allow_html=True)

    # Closing Note
    st.write("### 🚀 Stay Curious & Keep Exploring!")
    st.write("💡 Have feedback or suggestions? We'd love to hear from you! Happy exploring! 🎭")

if page == "Back to Home":
    st.markdown('<meta http-equiv="refresh" content="0;URL=http://127.0.0.1:5000/">', unsafe_allow_html=True)
elif page == "Introduction":
    st.title("Welcome to the Emotion Predictor App!")
    st.write("### About This App")
    st.write("""
        The **Emotion Predictor** app leverages machine learning models to analyze user inputs 
        and predict their dominant emotion. The app uses multiple pre-trained models and a weighted 
        voting system to deliver accurate predictions based on user interactions on different platforms.
    """)
    st.write("### Key Features")
    st.write("""
    - **Multiple Models**: Includes Random Forest, KNN, Decision Tree, Logistic Regression, and SVM.
    - **Weighted Voting**: Combines model predictions based on their accuracy to provide the most reliable outcome.
    - **Custom Inputs**: Enter personalized data such as age, gender, platform, and daily usage statistics to get tailored predictions.
    - **Prediction History**: View past predictions along with the input data for further insights.
    - **Visual Insights**: Understand the relationship between emotions and different user behaviors (e.g., daily usage, likes received).
    """)
    st.write("### How to Use")
    st.write("""
    1. Navigate to the **Emotion Predictor** page from the sidebar to enter your details.
    2. Provide information such as age, gender, platform, and engagement metrics like daily usage time and posts per day.
    3. Click on **Predict Emotion** to see the results based on your input data and the model’s predictions.
    4. Explore the **Prediction History** page to review past predictions and their corresponding input data.
    5. Visit the **Exploratory Data Analysis (EDA)** page to analyze visual insights and trends in emotion distribution across different user demographics.
    """)
    st.write("### Platforms Supported")
    st.write("""
    The **Emotion Predictor** app supports the following platforms:
    - **Instagram**: High emotional engagement with strong expressions of **Happiness** and **Sadness**.
    - **Twitter**: Characterized by **Anger** and **Sadness**, indicating strong emotional reactions.
    - **LinkedIn**: Shows moderate levels of **Boredom** and **Sadness**, reflecting a more neutral and less engaging experience.
    - **Facebook**: A balanced platform with **Neutral** and **Sadness** as dominant emotions.
    - **Snapchat**, **WhatsApp**, and **Telegram**: Show mixed emotional states with a balance of **Neutral**, **Anxiety**, and **Anger**.
    """)
    st.write("---")
    st.write("Get started by selecting **Emotion Predictor** from the sidebar to predict emotions based on your unique inputs!")


elif page == "Emotion Predictor":
    st.title("Emotion Predictor")
    st.write("Predict emotions using pre-trained models and weighted voting.")
    st.sidebar.title("User Inputs")

    # Add more input features
    age = st.sidebar.number_input("Age", min_value=0, max_value=35, step=1, value=25)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary", "Unknown"])
    platform = st.sidebar.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "Snapchat", "LinkedIn", "WhatsApp"])
    daily_usage_time = st.sidebar.slider("Daily Usage Time (minutes)", min_value=0, max_value=200, step=1, value=60)
    posts_per_day = st.sidebar.number_input("Posts Per Day", min_value=0.0, step=0.1, value=2.0)
    likes_received_per_day = st.sidebar.number_input("Likes Received Per Day", min_value=0.0, step=1.0, value=50.0)
    comments_received_per_day = st.sidebar.number_input("Comments Received Per Day", min_value=0.0, step=1.0, value=10.0)
    messages_sent_per_day = st.sidebar.number_input("Messages Sent Per Day", min_value=0.0, step=1.0, value=20.0)

    # Prediction button
    if st.button("Predict Emotion"):
        try:
            # Prepare input DataFrame
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Platform': [platform],
                'Daily_Usage_Time (minutes)': [daily_usage_time],
                'Posts_Per_Day': [posts_per_day],
                'Likes_Received_Per_Day': [likes_received_per_day],
                'Comments_Received_Per_Day': [comments_received_per_day],
                'Messages_Sent_Per_Day': [messages_sent_per_day]
            })
            
            # Get predictions from models
            predictions = {model_name: model.predict(input_data)[0] for model_name, model in models.items()}
            
            # Weighted voting logic
            weighted_predictions = {}
            for model_name, prediction in predictions.items():
                weight = model_weights.get(model_name, 1)
                if prediction not in weighted_predictions:
                    weighted_predictions[prediction] = 0
                weighted_predictions[prediction] += weight
            
            # Determine the most voted emotion
            most_voted_emotion = max(weighted_predictions, key=weighted_predictions.get)
            
            # Save prediction and inputs to database
            cursor.execute("""
            INSERT INTO history (timestamp, input_data, predicted_emotion)
            VALUES (?, ?, ?)
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), input_data.to_json(), most_voted_emotion))
            conn.commit()

            # Display results
            st.write(f"### Predicted Dominant Emotion (Weighted Voting): {most_voted_emotion}")
            st.write("### Model Predictions with Weights:")
            for name, prediction in predictions.items():
                weight = model_weights.get(name, 1)
                st.write(f"- {name}: {prediction} (Weight: {weight})")
            
            # Display prediction distribution chart
            st.subheader("Prediction Distribution")
            labels = list(weighted_predictions.keys())
            values = list(weighted_predictions.values())
            
            fig, ax = plt.subplots()
            ax.bar(labels, values, color='skyblue')
            ax.set_ylabel('Weight')
            ax.set_title('Emotion Voting Distribution')
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Visualizing the exploratory data analysis results for the dataset.")
    
    # Define the directory where your images are stored
    image_directory = r"C:\Users\Kiran\OneDrive\Desktop\DBMS LAB\aiml lab"

    # List the image files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]
    sorted_image_files = ["daily_usage_time_vs_dominant_emotion.png", 
                          "likes_received_per_day_vs_dominant_emotion.png", 
                          "dominant_emotion_by_gender.png","dominant_emotion_by_platform.png","dominant_emotion_distribution.png"]
    if image_files:
        st.write("### EDA Graphs")
        for image_file in sorted_image_files:
            # Create the full path to the image
            image_path = os.path.join(image_directory, image_file)
            
            # Display the image in the Streamlit app
            st.image(image_path, caption=image_file, use_container_width=True)

           # Display insights based on the image
            if image_file == "daily_usage_time_vs_dominant_emotion.png":
                st.write("""
    **Insights:**
    1. **Daily Usage Time vs Dominant Emotion**:
    - **Happiness** is the dominant emotion with the highest daily usage time, peaking around **150 minutes**.
    - **Anxiety** and **Anger** have similar average usage times, indicating frequent user engagement.
    - **Sadness** shows moderate daily usage, suggesting it has a noticeable but lesser impact compared to happiness.
    - **Neutral** emotions are maintained, hinting at a balanced emotional state for certain periods.
    - **Boredom** has the lowest usage time, implying it's the least engaging emotional state.
    - **Overall**, positive and strong emotions like **happiness** and **anxiety** lead to higher engagement.
    """)

            elif image_file == "likes_received_per_day_vs_dominant_emotion.png":
                st.write("""
    2. **Likes Received Per Day vs Dominant Emotion**:
    - **Happiness** leads engagement: Posts associated with **Happiness** receive the highest average likes per day, significantly outperforming other emotions.
    - **Neutral** and **Sadness** clusters: Both **Neutral** and **Sadness** show comparable average likes per day, suggesting a steady but moderate level of engagement.
    - **Anxiety** and **Anger** perform similarly: These emotions receive a similar number of likes, higher than **Boredom**, but much lower than **Happiness**.
    - Low engagement for **Boredom**: Posts expressing **Boredom** have the least engagement compared to all other emotions.
    """)

            elif image_file == "dominant_emotion_by_gender.png":
                st.write("""
    3. **Dominant Emotion by Gender**:
    - **Females**: The dominant emotion is **Happiness** (102 users), while other emotions (Anger, Neutral, Anxiety, and Sadness) have similar counts (56-48). **Boredom** has the least count (30).
    - **Males**: The dominant emotion is **Happiness** (66 users), with other emotions (Anger, Neutral, Anxiety, Boredom, and Sadness) showing similar counts (58-46).
    - **Non-binary**: The dominant emotion is **Neutral** (82 users). Other emotions (Anxiety, Sadness, and Boredom) have equal counts (46), while **Anger** is the least common (10).
    - **Unknown**: This group has the lowest overall emotional count, due to fewer data points.
    - Across all genders, **Happiness** leads for **Females** and **Males**, while **Neutral** is prominent for **Non-binary** individuals.
    """)

            elif image_file == "dominant_emotion_by_platform.png":
                st.write("""
    4. **Dominant Emotion by Platform**:
    - **Instagram** is the top platform for emotional engagement, particularly with high levels of **Happiness** and **Sadness**.
    - **Twitter** experiences a lot of **Anger** and **Sadness**, indicating strong emotional reactions from its users.
    - **LinkedIn** displays a notable amount of **Boredom** and **Sadness**, suggesting less engaging content for users.
    - **Facebook** has a balanced mix, but **Neutral** and **Sadness** are prominent, reflecting a varied user experience.
    - **Snapchat, WhatsApp, and Telegram** show lower emotional counts, with a blend of **Neutral**, **Anxiety**, and **Anger**.
    - **Overall**, platforms like **Instagram** and **Twitter** drive high emotional responses, while **LinkedIn** and **Facebook** lean towards more neutral or less engaging experiences.
    """)
            elif image_file == "dominant_emotion_distribution.png":
                 st.write("""
    5. **Dominant Emotion Distribution**:
    - **Happiness** is the most dominant emotion, accounting for **20.1%** of the total emotional distribution.
    - **Neutral** emotions make up a significant portion at **20%**, reflecting a balanced emotional state among users.
    - **Anxiety** is notable, representing **17%** of the emotional distribution, indicating a common feeling among users.
    - **Sadness** is present in **16%** of the cases, showing a considerable amount of users experiencing this emotion.
    - **Boredom** accounts for **14%**, suggesting some level of disinterest or lack of engagement among users.
    - **Anger** is the least represented emotion at **13%**, but still noteworthy as it affects a portion of the user base.
    """)
                 
    else:
        st.write("No images found in the specified directory.")

elif page == "Prediction History":
    st.title("Prediction History")
    st.write("View past predictions along with the associated input data.")
    
    # Option to delete all history
    if st.button("Delete All History"):
        # Confirm deletion with the user
        confirm = st.confirm("Are you sure you want to delete all prediction history?")
        if confirm:
            cursor.execute("DELETE FROM history")
            conn.commit()
            st.success("All prediction history has been deleted.")
    
    # Fetch data from the database
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            timestamp = row[1]
            input_data = pd.read_json(row[2])
            predicted_emotion = row[3]

            # Display prediction history in a formatted way
            st.write(f"### Timestamp: {timestamp}")
            st.write("#### Input Data:")
            for col in input_data.columns:
                st.write(f"- **{col}**: {input_data.iloc[0][col]}")
            st.write(f"#### Predicted Emotion: **{predicted_emotion}**")
            st.write("---")
    else:
        st.write("No prediction history available.")

