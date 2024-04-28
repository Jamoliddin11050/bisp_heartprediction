from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle
from sklearn.preprocessing import StandardScaler
import psycopg2
import os

app = Flask(__name__)
app.secret_key = '6c8abf78a416c7369f2f6faaaff0068d'  # Set a secret key for session management

# Load the trained model and scaler
with open('logistic_regression_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(
        host="localhost",  # This should be 'localhost' if running on the same machine
        database="Machine",  # Ensure the database name is correct
        user="postgres",  # Default PostgreSQL user, ensure it's enabled and has the right permissions
        password="jamol"  # Change 'jamol' to the actual password for the 'postgres' user
    )
except psycopg2.OperationalError as e:
    print("Unable to connect to the database:", e)
    raise e  # Optionally, raise the error to stop execution if the database is essential

# Create a cursor object
cur = conn.cursor()

# Create users table if not exists
cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(100) NOT NULL
    )
''')
conn.commit()

# Create user_data table if not exists
cur.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(id),
        age INT,
        sex VARCHAR(10),
        cigsPerDay INT,
        totChol INT,
        sysBP FLOAT,
        glucose FLOAT
    )
''')
conn.commit()

# Insert user input data into the user_data table and automatically generate user_id
def insert_user_data(input_data):
    cur.execute('''
        INSERT INTO user_data (user_id, age, sex, cigsPerDay, totChol, sysBP, glucose)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    ''', input_data)
    user_id = cur.fetchone()[0]  # Fetch the generated user_id
    conn.commit()
    return user_id

# Retrieve user input history from the database
def get_user_data(user_id=None):
    if user_id is None:
        cur.execute('''SELECT * FROM user_data ORDER BY id DESC''')
    else:
        cur.execute('''SELECT * FROM user_data WHERE user_id=%s ORDER BY id DESC''', (user_id,))
    user_data = cur.fetchall()
    return user_data

# Prediction function
def predict(input_data):
    # Preprocess the input data
    input_data_scaled = scaler.transform([input_data])
    # Predict
    prediction = model.predict(input_data_scaled)
    return prediction.tolist()
@app.route('/')
def main_page():
    return render_template('main_page.html')
# Registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if username already exists
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cur.fetchone()
        if existing_user:
            return render_template('register.html', error="Username already exists. Please choose another one.")
        else:
            # Insert new user into the database
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
    else:
        # Render registration form template
        return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if username and password match
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        if user:
            session['username'] = username  # Storing username in session
            return redirect(url_for('index'))  # Redirect to index (form filling) page
        else:
            return render_template('login.html', error="Invalid username or password. Please try again.")
    else:
        # Render login form template
        return render_template('login.html')

# Form-filling page
@app.route('/index', methods=['GET', 'POST'])
def index():
    username = session.get('username')
    if username:
        if request.method == 'POST':
            input_data = [
                request.form['age'],
                request.form['sex'],
                request.form['cigsPerDay'],
                request.form['totChol'],
                request.form['sysBP'],
                request.form['glucose']
            ]
            user_id = insert_user_data(input_data + [username])  # Automatically generate user_id
            return redirect(url_for('dashboard'))  # Redirect to the dashboard
        return render_template('index.html', username=username)
    else:
        return redirect(url_for('login'))

import requests

# Function to fetch global health data from an API
def fetch_global_health_data():
    # Example API endpoint for fetching global health data
    api_url = "https://api.globalhealth5050.org/api/v1/summary"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        global_health_data = response.json()
        return global_health_data
    except requests.exceptions.RequestException as e:
        print("Failed to fetch global health data:", e)
        return None

# Dashboard page
@app.route('/dashboard')
def dashboard():
    username = session.get('username')
    if username:
        user_data = get_user_data()  # Retrieve user data for the dashboard
        global_health_data = fetch_global_health_data()  # Fetch global health data
        if global_health_data:
            return render_template('dashboard.html', username=username, user_data=user_data, global_health_data=global_health_data)
        else:
            return render_template('dashboard.html', username=username, user_data=user_data, error="Failed to fetch global health data")
    else:
        return redirect(url_for('login'))  # Redirect to login page if not logged in


# API endpoint to receive input data and return prediction
@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json()
    input_data = [data[key] for key in sorted(data.keys())]
    prediction = predict(input_data)
    user_data = None
    return jsonify({'prediction': prediction, 'user_data': user_data})
@app.route('/donate')
def donate():
    return render_template('donate.html')


if __name__ == '__main__':
    app.run(debug=True)
