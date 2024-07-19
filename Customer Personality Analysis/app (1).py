from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        income = float(request.form['Income'])
        recency = float(request.form['Recency'])
        mnt_wines = float(request.form['MntWines'])
        mnt_fruits = float(request.form['MntFruits'])
        mnt_meat_products = float(request.form['MntMeatProducts'])
        mnt_fish_products = float(request.form['MntFishProducts'])
        mnt_sweet_products = float(request.form['MntSweetProducts'])
        mnt_gold_prods = float(request.form['MntGoldProds'])
        num_deals_purchases = float(request.form['NumDealsPurchases'])
        num_web_purchases = float(request.form['NumWebPurchases'])
        num_catalog_purchases = float(request.form['NumCatalogPurchases'])
        num_store_purchases = float(request.form['NumStorePurchases'])
        num_web_visits_month = float(request.form['NumWebVisitsMonth'])
        age = float(request.form['Age'])

        features = np.array([[income, recency, mnt_wines, mnt_fruits, mnt_meat_products, mnt_fish_products,
                              mnt_sweet_products, mnt_gold_prods, num_deals_purchases, num_web_purchases,
                              num_catalog_purchases, num_store_purchases, num_web_visits_month, age]])

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        cluster_descriptions = {
            0: 'Cluster 0: Moderate income, moderate spending, moderate age.',
            1: 'Cluster 1: High income, high spending, older age.',
            2: 'Cluster 2: Low income, low spending, younger age.',
            3: 'Cluster 3: Very high income, very high spending, very old age.'
        }
        
        prediction_text = cluster_descriptions[prediction]
        
        return render_template('index.html', prediction_text=prediction_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
