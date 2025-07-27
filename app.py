import streamlit as st
import pandas as pd

import joblib
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import os
import collections
import json

# Helper to load model safely
def safe_load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Model file '{filename}' not found or could not be loaded: {e}")
        return None

# Load pre-trained models (relative paths)
rf_pipeline = safe_load_model(os.path.join('pkl', 'rf_pipeline.pkl'))
knn_pipeline = safe_load_model(os.path.join('pkl', 'knn_pipeline.pkl'))
dt_pipeline = safe_load_model(os.path.join('pkl', 'dt_pipeline.pkl'))
lr_pipeline = safe_load_model(os.path.join('pkl', 'lr_pipeline.pkl'))
svm_pipeline = safe_load_model(os.path.join('pkl', 'svm_pipeline.pkl'))
stacking_model = safe_load_model(os.path.join('pkl', 'stacking_model.pkl'))


# Pre-trained models (skip None)
models = {}
if rf_pipeline: models['RandomForest'] = rf_pipeline
if knn_pipeline: models['KNN'] = knn_pipeline
if dt_pipeline: models['DecisionTree'] = dt_pipeline
if lr_pipeline: models['LogisticRegression'] = lr_pipeline
if svm_pipeline: models['SVM'] = svm_pipeline
if stacking_model: models['Stacking'] = stacking_model

# Weights for each model based on accuracy
model_weights = {
    'RandomForest': 5,
    'KNN': 4,
    'DecisionTree': 3,
    'SVM': 3,
    'LogisticRegression': 2,
    'Stacking': 6
}

def create_database():
    with sqlite3.connect("predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE,
                          password TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT,
                          text TEXT,
                          prediction TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT,
                          timestamp TEXT,
                          input_data TEXT,
                          predicted_emotion TEXT)''')
        conn.commit()

def authenticate(username, password):
    with sqlite3.connect("predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        return cursor.fetchone() is not None

def register_user(username, password):
    with sqlite3.connect("predictions.db") as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def save_prediction(username, text, prediction):
    with sqlite3.connect("predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (username, text, prediction) VALUES (?, ?, ?)",
                       (username, text, prediction))
        conn.commit()

def get_prediction_history(username):
    with sqlite3.connect("predictions.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text, prediction FROM predictions WHERE username=?", (username,))
        return cursor.fetchall()

def reset_password(username, new_password):
        with sqlite3.connect("predictions.db") as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
            if cursor.rowcount == 0:
                return False
            conn.commit()
            return True
                    
# Streamlit UI Setup
st.set_page_config(page_title="Emotion Predictor", layout="centered")

# Create database
create_database()

# Sidebar navigation
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

if not st.session_state["logged_in"]:
    st.warning("🚫 Please log in to access the features!")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()

        else:
            st.error("Invalid username or password")
    
    st.write("Don't have an account?")
    if st.button("Register"):
        st.session_state["show_register"] = True

    if st.session_state.get("show_register", False):
        st.title("Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if register_user(new_username, new_password):
                st.success("Account created successfully! Please login.")
                st.session_state["show_register"] = False
            else:
                st.error("Username already exists. Please choose a different username.")
    
    # Add the Reset Password option
    st.write("Forgot your password?")
    if st.button("Reset Password"):
        st.session_state["show_reset"] = True
    
    if st.session_state.get("show_reset", False):
        st.title("Reset Password")
        # Use unique keys to avoid conflicts with login fields
        reset_username = st.text_input("Enter your Username", key="reset_username")
        new_password = st.text_input("Enter new Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm new Password", type="password", key="confirm_password")
        if st.button("Submit Password Reset"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                # reset_password() should be a function that handles updating the user's password.
                if reset_password(reset_username, new_password):
                    st.success("Password reset successfully! Please log in with your new password.")
                    st.session_state["show_reset"] = False
                else:
                    st.error("Username not found or error resetting password. Please try again.")
    
# Check login before allowing access to pages
if not st.session_state["logged_in"]:
    st.stop()  # Prevents further execution

# If logged in, proceed with navigation
page = st.sidebar.radio("Navigation", ["Introduction", "Emotion Predictor", "EDA", "Prediction History", "Back to Home"])

if page == "Back to Home":
    st.title("🎉 Thank You for Exploring Emotion Predictor!")
    st.write("""
        We truly appreciate your time in exploring our Emotion Predictor Platform.  
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
    st.write("🔎 *Explore different sections from the Navigation Bar before heading back home!*")

    # Closing Note
    st.write("### 🚀 Stay Curious & Keep Exploring!")
    st.write("💡 Have feedback or suggestions? Mail us: kiranraithal.cd22@rvce.edu.in We'd love to hear from you! Happy exploring! 🎭")
else:
    if page == "Introduction":
        st.title("Welcome to the Emotion Predictor App!")
        st.markdown(f"<h2 style='color: #2E86C1;'>Hello, {st.session_state['username']}! </h2>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()
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
        platform = st.sidebar.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "Snapchat", "LinkedIn", "Telegram", "WhatsApp"])
        daily_usage_time = st.sidebar.slider("Daily Usage Time (minutes)", min_value=0, max_value=200, step=1, value=60)
        posts_per_day = st.sidebar.number_input("Posts Per Day", min_value=0, step=1, value=2)
        likes_received_per_day = st.sidebar.number_input("Likes Received Per Day", min_value=0, step=1, value=50)
        comments_received_per_day = st.sidebar.number_input("Comments Received Per Day", min_value=0, step=1, value=10)
        messages_sent_per_day = st.sidebar.number_input("Messages Sent Per Day", min_value=0, step=1, value=20)

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
                with sqlite3.connect("predictions.db") as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                INSERT INTO history (username,timestamp, input_data, predicted_emotion)
                VALUES (?, ?, ?, ?)
                """, (st.session_state["username"],datetime.now().strftime("%Y-%m-%d %H:%M:%S"), input_data.to_json(), most_voted_emotion))
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
    
    # Define the directory where your images are stored (relative for Streamlit Cloud)
        image_directory = "images"
        if not os.path.exists(image_directory):
            st.warning(f"Image directory '{image_directory}' not found. Please add an 'images' folder with your .png/.jpg files.")
            image_files = []
        else:
            image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]

        sorted_image_files = [
            "daily_usage_time_vs_dominant_emotion.png",
            "likes_received_per_day_vs_dominant_emotion.png",
            "dominant_emotion_by_gender.png",
            "dominant_emotion_by_platform.png",
            "dominant_emotion_distribution.png",
            "Post_Per_Day_By_Gender.png"
        ]
        if image_files:
            st.write("### EDA Graphs")
            for image_file in sorted_image_files:
                image_path = os.path.join(image_directory, image_file)
                if os.path.exists(image_path):
                    st.image(image_path, caption=image_file, use_container_width=True)
                else:
                    st.warning(f"Image '{image_file}' not found in '{image_directory}'.")

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
                elif image_file == "Post_Per_Day_By_Gender.png":
                    st.write("""
    6. **Post Per Day By Gender**:
    - **Instagram**: This platform sees the highest number of posts per day, predominantly from Female users, followed by Male users, with smaller contributions from Non-binary and Unknown users.
    - **Twitter**: It ranks second in terms of daily posts, with a majority of the content coming from Female users and a smaller portion from Male users.
    - **Facebook**: Displays a balanced distribution of posts among Male, Female, and Non-binary users, with a small fraction from Unknown users.
    - **Snapchat and Whatsapp**: These platforms exhibit a similar trend, where Female users contribute the most posts, followed by Male users, and smaller contributions from Non-binary and Unknown users.
    - **LinkedIn and Telegram**: These platforms have the lowest number of posts per day. LinkedIn shows a more balanced distribution among all genders, while Telegram sees a higher contribution from Male users and smaller portions from Non-binary and Unknown users.
    """)
        
        else:
            st.write("No images found in the specified directory.")

    elif page == "Prediction History":
        st.title("Prediction History")
    # Option to delete all history for the logged-in user
        if st.button("Delete All History"):
        # Confirm deletion with the user
            confirm = st.confirm("Are you sure you want to delete all prediction history?")
            if confirm:
                with sqlite3.connect("predictions.db") as conn:
                    cursor = conn.cursor()
                # Delete only history for the current user
                    cursor.execute("DELETE FROM history WHERE username=?", (st.session_state["username"],))
                    conn.commit()
                    st.success("All prediction history has been deleted.")

    # Fetch data from the database for the logged-in user
        with sqlite3.connect("predictions.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM history WHERE username=? ORDER BY timestamp DESC", (st.session_state["username"],))
            rows = cursor.fetchall()
    
        if rows:
            from collections import Counter
            import matplotlib.pyplot as plt
        
            # Aggregate predicted emotions for a pie chart
            emotion_counts = Counter()
            for row in rows:
            # Assuming the predicted emotion is in column index 4
                emotion_counts[row[4]] += 1
        
        # Create and display a pie chart for dominant emotions distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title("Dominant Emotions Distribution")
            st.pyplot(fig)
            # Provide recommendations based on the most frequent emotion
            if emotion_counts:
                most_common_emotion, count = emotion_counts.most_common(1)[0]
                st.write(f"### Most Frequent Emotion: {most_common_emotion} (Count: {count})")
                if most_common_emotion.lower() == "angry":
                    st.info("It appears you often feel angry. Consider taking a short break, practicing deep breathing, or engaging in physical activity to help release tension.")
                elif most_common_emotion.lower() == "sadness":
                    st.info("It seems you are frequently sad. Consider activities that boost your mood such as talking to a friend, listening to uplifting music, or engaging in a hobby you love.")
                elif most_common_emotion.lower() == "happiness":
                    st.info("You're mostly happy! Keep up your positive habits and continue sharing your joy with those around you.")
                elif most_common_emotion.lower() == "neutral":
                    st.info("Your emotional state is mostly neutral. Consider exploring new activities or hobbies that may add more excitement to your day.")
                elif most_common_emotion.lower == "boredom":
                    st.info("It appears you frequently feel bored. Consider exploring new hobbies, learning new skills, or engaging in creative activities to add more excitement to your day.")
                elif most_common_emotion.lower() == "anxiety":
                    st.info("You seem to experience anxiety often. Consider mindfulness exercises, meditation, or talking to a professional if you feel overwhelmed.")
                st.write(f"""
    Based on your historical data, your predominant emotional state is **{most_common_emotion}**. 
    This insight suggests that you may benefit from tailored strategies to enhance your overall well-being. 
    Regularly reviewing your emotional trends can help you recognize patterns and adjust your daily activities accordingly—whether that means incorporating more relaxation techniques, engaging in mood-boosting activities, or seeking professional support when needed.
    """)
            for row in rows:
                timestamp = row[2]
                json_str = row[3].strip()
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                # Attempt to trim extra data by taking substring up to the last closing brace/bracket
                    if json_str and json_str[0] == '{':
                        idx = json_str.rfind('}')
                        json_str_trimmed = json_str[:idx+1]
                    elif json_str and json_str[0] == '[':
                        idx = json_str.rfind(']')
                        json_str_trimmed = json_str[:idx+1]
                    else:
                        st.error(f"Error decoding JSON: {e}")
                        continue
                    try:
                        data = json.loads(json_str_trimmed)
                    except Exception as e2:
                        st.error(f"Error decoding JSON after trimming: {e2}")
                        continue
            
            # Convert data into a list of dictionaries if it's in dict format (default from DataFrame.to_json())
                if isinstance(data, dict):
                # The default orientation is {column: {index: value}}, so convert it to a list of row dictionaries
                    try:
                    # Assume there's only one row of data
                        row_dict = {col: list(data[col].values())[0] for col in data}
                        data = [row_dict]
                    except Exception as conv_error:
                        st.error(f"Error converting JSON data to DataFrame format: {conv_error}")
                        continue
                elif not isinstance(data, list):
                # If it's not a list, wrap it in a list
                    data = [data]
            
                try:
                    input_data = pd.DataFrame(data)
                except Exception as e:
                    st.error(f"Error creating DataFrame: {e}")
                    continue
            
                predicted_emotion = row[4]
            # Display prediction history in a formatted way
                st.write(f"### Timestamp: {timestamp}")
                st.write("#### Input Data:")
                for col in input_data.columns:
                    st.write(f"- **{col}**: {input_data.iloc[0][col]}")
                st.write(f"#### Predicted Emotion: **{predicted_emotion}**")
                st.write("---")
        else:
            st.write("No prediction history available.")