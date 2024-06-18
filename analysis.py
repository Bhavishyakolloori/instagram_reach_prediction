import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data

# Plotting functions
def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Likes'], data['Impressions'], alpha=0.5)
    plt.title('Likes vs Impressions')
    plt.xlabel('Likes')
    plt.ylabel('Impressions')
    plt.savefig('static/likes_vs_impressions.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.scatter(data['Comments'], data['Impressions'], alpha=0.5)
    plt.title('Comments vs Impressions')
    plt.xlabel('Comments')
    plt.ylabel('Impressions')
    plt.savefig('static/comments_vs_impressions.png')
    plt.clf()

# Feature Engineering and Model Building
def build_model(data):
    features = data[['Likes', 'Comments', 'Shares', 'Saves', 'Profile Visits']]
    target = data['Impressions']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2

# Prediction function
def predict_impressions(model, likes, comments, shares, saves, profile_visits):
    input_data = pd.DataFrame({
        'Likes': [likes],
        'Comments': [comments],
        'Shares': [shares],
        'Saves': [saves],
        'Profile Visits': [profile_visits]
    })
    return model.predict(input_data)[0]
