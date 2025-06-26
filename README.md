# A/B Testing to Compare Performance of two ML Models in Production
In this project, we deploy Machine Learning Models A (Logistic Regression) and B (Random Forest) to a simple Flask-based API behind an Nginx Reverse Proxy that routes traffic 50/50 to each model, simulating A/B testing. We will log the predictions and user feedback to parse it and analyze which model performs better based on our users (a.k.a customers).


## Requirements
1. Python:                  For Building the Classification Model 
2. Flask:                   To Expose the ML Model Inference API
3. Nginx Reverse Proxy:     For Traffic Load Balancing between the 2 Models
4. Docker & Docker-compose: For Containerization and Service Orchestration
5. Streamlit:               For Visualizing Model Feedback Performance


## Project Structure
Create the required project directories
```sh
mkdir models api nginx logs
```


## Create Python Virtual ENV
```sh
python3 -m venv ab-test-venv
source ab-test-venv/bin/activate
```

## Install Required Python Packages

```sh
cd api
touch requirements.txt
```

Add the following to requirements.txt
```sh
# requirements.txt
flask
scikit-learn
numpy
pandas
streamlit
```

```sh
pip install -r requirements.txt
```


## Model Scripts

### Create ML Model Training Scripts
```sh
cd models
touch train_model_a.py train_model_b.py
``` 

Add the codes below for Models a & b: 
```sh
# train_model_a.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

with open("model_a.pkl", "wb") as f:
    pickle.dump(model, f)
```

```sh
# train_model_b.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open("model_b.pkl", "wb") as f:
    pickle.dump(model, f)
```

### Train the ML Models
```sh
cd models
python3 train_model_a.py
python3 train_model_b.py
```


## Create Flask APIs for Both Models
```sh
cd api
touch app_model_a.py app_model_b.py
```

```sh
# app_model_a.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
import datetime
import os

app = Flask(__name__)
model = pickle.load(open("model_a.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    feedback = request.json.get('feedback', None)
    prediction = model.predict([np.array(data)])
    
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "model": "ModelA",
        "features": data,
        "prediction": int(prediction[0]),
        "feedback": feedback
    }
    features_str = ";".join(map(str, data))
    with open("/logs/predictions.log", "a") as f:
        f.write(f"{datetime.datetime.now()},ModelA,{features_str},{int(prediction[0])},{feedback}\n")
    
    return jsonify({'model': 'A', 'prediction': int(prediction[0]), 'logged': True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

```sh
# app_model_b.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
import datetime
import os

app = Flask(__name__)
model = pickle.load(open("model_b.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    feedback = request.json.get('feedback', None)
    prediction = model.predict([np.array(data)])
    
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "model": "ModelB",
        "features": data,
        "prediction": int(prediction[0]),
        "feedback": feedback
    }

    features_str = ";".join(map(str, data))
    with open("/logs/predictions.log", "a") as f:
        f.write(f"{datetime.datetime.now()},ModelB,{features_str},{int(prediction[0])},{feedback}\n")
    
    return jsonify({'model': 'B', 'prediction': int(prediction[0]), 'logged': True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```


## Setup NGINX as a 50/50 Traffic Splitter
By default, NGINX distributes requests equally (round-robin) among servers listed in an upstream block.

```sh
cd nginx
touch default.conf
```

```sh
# default.conf
events {}

http {
    upstream backend {
        server model_a:5000 weight=1;
        server model_b:5000 weight=1;
    }

    server {
        listen 80;

        location /predict {
            proxy_pass http://backend;
        }
    }
}
```


## Containerize the Flask-based ML Model APIs with Dockerfile & Docker-Compose

### Go to project root directory and create the Dockerfiles
```sh
touch Dockerfile.model_a Dockerfile.model_b
```

The Dockerfiles below are for both model_a and model_b. 

```sh
# Dockerfile.model_a

FROM python:3.10-slim

WORKDIR /app

COPY api/ ./
COPY models/model_a.pkl ./model_a.pkl

RUN pip install -r requirements.txt

CMD ["python", "app_model_a.py"]
```

```sh
# Dockerfile.model_b

FROM python:3.10-slim

WORKDIR /app

COPY api/ ./
COPY models/model_b.pkl ./model_b.pkl

RUN pip install -r requirements.txt

CMD ["python", "app_model_b.py"]
```


### Use Docker Compose to Orchestrate Building the Images
With the Dockerfiles ready in the project root directory, create your docker-compose.yml file in the project root that will:

1. Build the images for both models (one for model_a, and the other for model_b) with the Dockerfiles
2. Install the required Python packages.
3. Use different Flask app entry points using the "command" override.
4. Use NGINX to route traffic 50/50 between them.


```sh
touch docker-compose.yml
```

```sh
# docker-compose.yml
version: '3'

services:
  model_a:
    build:
      context: .
      dockerfile: Dockerfile.model_a
    container_name: model_a
    volumes:
      - ./logs:/logs
    ports:
      - "5001:5000"

  model_b:
    build:
      context: .
      dockerfile: Dockerfile.model_b
    container_name: model_b
    volumes:
      - ./logs:/logs
    ports:
      - "5002:5000"

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx/default.conf:/etc/nginx/nginx.conf
```

### Run docker-compose.yml from Project Root
```sh
docker-compose up --build
```

To stop docker-compose, run the command
```sh
docker-compose down
```


## Let's Create Some Inference/Prediction Requests to the /predict API Endpoint

```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.2, 3.4, 5.4, 2.3], "feedback": true}'
```

```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.2, 3.4, 5.4, 2.3], "feedback": false}'
```

```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2], "feedback": true}'
```

```sh
curl -X POST http://localhost:8080/predict  \
  -H "Content-Type: application/json" \
  -d '{"features": [2.7, 4.7, 4.1, 2.3], "feedback": true}'
```


```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2], "feedback": false}'
```


## Analyze A/B Test Results
Now, run a script to determine which model is performing better.
In the project root directory, create analyze_ab_test.py file to analyze the A/B test results. 
```sh
touch analyze_ab_test.py
```

```sh
# analyze_ab_test.py

import pandas as pd

df = pd.read_csv(
    "./logs/predictions.log",
    header=None,
    names=["timestamp", "model", "features", "prediction", "feedback"]
)

df["feedback"] = df["feedback"].astype(str).str.lower() == "true"
df = df[df["feedback"].notnull()]  # Only include rows where feedback was given

results = df.groupby("model")["feedback"].mean()
print("\nModel Accuracy Based on User Feedback:")
print(results)
```


## Visualize Model Performance in Streamlit
This Streamlit dashboard will show:

1. Number of predictions per model
2. Accuracy per model (based on feedback)
3. Line chart of prediction accuracy over time

```sh
touch streamlit_app.py
```

```sh
# streamlit_app.py

import pandas as pd
import streamlit as st

st.set_page_config(page_title="A/B Test Dashboard", layout="wide")

st.title("📊 A/B Testing Results for ML Models")

# Load log file
try:
    df = pd.read_csv("logs/predictions.log", header=None,
                     names=["timestamp", "model", "features", "prediction", "feedback"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["feedback"] = df["feedback"].astype(str).str.lower() == "true"

    st.sidebar.subheader("Summary")
    st.sidebar.write(f"Total Predictions: {len(df)}")
    st.sidebar.write(f"With Feedback: {df['feedback'].notnull().sum()}")

    st.subheader("Prediction Counts")
    st.bar_chart(df["model"].value_counts())

    st.subheader("Accuracy by Model")
    acc = df.groupby("model")["feedback"].mean().sort_values(ascending=False)
    st.dataframe(acc.rename("Accuracy (%)") * 100)

    st.subheader("Accuracy Over Time (Optional)")
    df["date"] = df["timestamp"].dt.date
    daily_acc = df.groupby(["date", "model"])["feedback"].mean().unstack()
    st.line_chart(daily_acc)

except Exception as e:
    st.error(f"Failed to load data: {e}")
```


### Run the Streamlit App
From the project root, run:
```sh
streamlit run streamlit_app.py
```

Next, open the Streamlit App in your browser to view your real-time A/B testing dashboard!
```sh
http://localhost:8501
```


**Please Like, Comment, and Subscribe to iQuant on YouTube**