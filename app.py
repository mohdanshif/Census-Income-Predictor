from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

with open('classification_model.pkl','rb') as f:
    model=pickle.load(f)
    
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)   
    
@app.route('/')
def home():
    return render_template('model.html')

@app.route('/prediction', methods=["post"])
def index():
    # Get form values
    marital_status = request.form.get('marital-status')
    occupation = request.form.get('occupation')

    # Create one-hot encoding based on values
    marital_status_married = 1 if marital_status == 'Married-civ-spouse' else 0
    occupation_sales = 1 if occupation == 'Sales' else 0

    # Build input data
    input_data = {
        'age': int(request.form['age']),
        'education-num': int(request.form['education-num']),
        'capital-gain': float(request.form['capital-gain']),
        'capital-loss': float(request.form['capital-loss']),
        'hours-per-week': int(request.form['hours-per-week']),
        'marital-status_Married-civ-spouse': marital_status_married,
        'occupation_Sales': occupation_sales,
    }

    input_df = pd.DataFrame([input_data])

    # Standardize and transform
    num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])
    input_df['age'] = np.log1p(input_df['age'])

    predicted_class = model.predict(input_df)
    predict = "Predicted income is above 50k" if predicted_class == 1 else "Predicted income is below 50k"
    
    return render_template('model.html', prediction=predict)
if __name__=='__main__':
     app.run(debug=True)



