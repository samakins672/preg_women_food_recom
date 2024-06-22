from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved components
with open('extracted_data.pkl', 'rb') as f:
    extracted_data = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('neigh.pkl', 'rb') as f:
    neigh = pickle.load(f)

# Nutritional guidelines for pregnant women
max_list = [2500, 85, 20, 300, 2300, 350, 28, 50, 75]
ingredient_filter = ['swordfish', 'shark', 'king mackerel', 'tilefish', 'alcohol', 'caffeine']

def apply_pipeline(neigh, scaler, _input, extracted_data):
    prep_data = scaler.transform(extracted_data.iloc[:, 6:15].to_numpy())
    indices = neigh.kneighbors(_input, return_distance=False)[0]
    recommendations = extracted_data.iloc[indices]
    return list(zip(recommendations['Name'], recommendations['RecipeIngredientParts']))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        saturated_fat = float(request.form['saturated_fat'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        carbohydrate = float(request.form['carbohydrate'])
        fiber = float(request.form['fiber'])
        sugar = float(request.form['sugar'])
        protein = float(request.form['protein'])

        user_input = [[calories, fat, saturated_fat, cholesterol, sodium, carbohydrate, fiber, sugar, protein]]
        recommendations = apply_pipeline(neigh, scaler, user_input, extracted_data)

        return render_template('results.html', recommendations=recommendations)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
