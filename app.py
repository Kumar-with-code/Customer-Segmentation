from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, 
            template_folder='p-seg/templates', 
            static_folder='p-seg/static')

app = Flask(__name__)

# load model
model = pickle.load(open('kmeans_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    score = float(request.form['score'])

    data = np.array([[income, score]])
    cluster = model.predict(data)[0]


    # map cluster meaning
    if cluster == 1:
        result = " Premium Customer"
    elif cluster == 3:
        result = " Rich but Low Spender"
    elif cluster == 2:
        result = " Low Income High Spender"
    elif cluster == 4:
        result = " Low Value Customer"
    else:
        result = " Average Customer"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)