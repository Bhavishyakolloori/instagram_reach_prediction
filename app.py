from flask import Flask, request, render_template
import analysis

app = Flask(__name__)

# Load and process data
data = analysis.load_data('Instagram data.csv')
analysis.plot_data(data)
model, mae, r2 = analysis.build_model(data)

@app.route('/')
def home():
    return render_template('index.html', mae=mae, r2=r2)

@app.route('/predict', methods=['POST'])
def predict():
    likes = int(request.form['likes'])
    comments = int(request.form['comments'])
    shares = int(request.form['shares'])
    saves = int(request.form['saves'])
    profile_visits = int(request.form['profile_visits'])
    
    prediction = analysis.predict_impressions(model, likes, comments, shares, saves, profile_visits)
    return render_template('index.html', prediction=prediction, mae=mae, r2=r2)

if __name__ == '__main__':
    app.run(debug=True)
