from flask import Flask,request,jsonify
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "Tourist place recommender"

@app.route('/predict', methods=['POST'])
def predict():
    sea_lover = request.form.get('sea_lover')
    mountain_lover = request.form.get('mountain_lover')
    history_lover = request.form.get('history_lover')
    entertainment_lover = request.form.get('entertainment_lover')
    need_hotel = request.form.get('need_hotel')
    hotel_type = request.form.get('hotel_type')
    need_transport = request.form.get('need_transport')
    days = request.form.get('days')
    place = request.form.get('place')
    budget = request.form.get('budget')
    travel_guide = request.form.get('travel_guide')
    prefer_attractions = request.form.get('prefer_attractions')
    traveling_partner = request.form.get('traveling_partner')
    prefer_safety = request.form.get('prefer_safety')
    foodie = request.form.get('foodie')
    tourist_friendly_place = request.form.get('tourist_friendly_place')

    starting_point = request.form.get('starting_point')


    input_query=np.array([[sea_lover,mountain_lover,history_lover,entertainment_lover,need_hotel,hotel_type,need_transport,days,place,budget,travel_guide,prefer_attractions,traveling_partner,prefer_safety,foodie,tourist_friendly_place,starting_point]])
    result=model.predict(input_query)[0]

    return jsonify({'suitable trip':str(result)})



if __name__=='__main__':
    app.run(debug=True)