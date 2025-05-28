🧠 Income Prediction Using Census Data
Welcome to the Income Prediction ML Project!
In this project, we build a machine learning model to predict whether an individual earns more than $50K/year using demographic and employment-related data.

🎯 Objective
The main goals of this project are:

Perform Exploratory Data Analysis (EDA) on the "Adult Census Income" dataset.

Preprocess the data to handle missing values, encode categorical features, and scale numerical ones.

Build and evaluate a classification model to predict income (>50K or <=50K).

Deploy the model using:

A Flask web application

A Streamlit web application

📦 Dataset Information
Dataset Name: Adult Census Income

Source: OpenML - Dataset 1590

Loading Method:

python
Copy
Edit
from sklearn.datasets import fetch_openml
data = fetch_openml("adult", version=2, as_frame=True)
Target Column: income

Prediction Task: Binary Classification

🧪 Methodology
EDA:

Identified distributions of key variables like age, education, occupation, and income.

Analyzed correlation and class imbalances.

Preprocessing:

Handled missing values.

Applied label encoding and one-hot encoding.

Normalized numerical features using StandardScaler.

Model Building:

Trained multiple models including Logistic Regression, Random Forest, and Gradient Boosting.

Selected the best-performing model based on accuracy, precision, recall, and F1-score.

Model Saving:

Used pickle to save both the trained model and scaler for reuse in the web apps.

🚀 Deployment
🔧 Flask Web Application
Built using Python and Flask.

Provides a web form for users to input their data.

Displays predicted income category on form submission.

Run:

bash
Copy
Edit
python app.py
🎈 Streamlit Application
Interactive UI using Streamlit components.

Real-time prediction display with a clean interface.

Run:

bash
Copy
Edit
streamlit run streamlit.py
📂 Project Structure
graphql
Copy
Edit
├── eda_preprocessing.ipynb      # EDA and preprocessing notebook
├── model_training.ipynb         # Model building and evaluation
├── classification_model.pkl     # Trained classification model
├── scalar.pkl                   # Scaler used for standardization
├── app.py                       # Flask or Streamlit app (rename as needed)
├── templates/
│   └── form.html                # HTML form for Flask app
├── requirements.txt             # Required libraries
└── README.md                    # Project documentation
📷 Screenshots
(Insert screenshots here of the Streamlit/Flask app UI and sample predictions)

🌐 Live Demo (Optional)
Streamlit App: View Demo (Add your actual link if deployed)

Flask App: Run locally using python app.py

📌 Requirements
Install dependencies via:

bash
Copy
Edit
pip install -r requirements.txt
📬 Submission
Please push all files to your forked GitHub repository and submit the repo link.


income-predictor/
├── flask_app.py                # Flask app entry point
├── streamlit_app.py            # Streamlit app entry point
├── model/
│   └── model.pkl               # Trained and serialized model files
├── templates/
│   └── index.html              # HTML template for Flask form
├── static/                     # (Optional) Static files for Flask
├── notebook/
│   └── eda\_model.ipynb        # notebooks
├── requirements.txt            # List of Python dependencies
└── README.md                   # Updated with your project details
```
