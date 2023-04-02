from flask import Flask, render_template, request
import numpy as np
import pickle

#bitcoin_model = pickle.load(open('Models/bitcoinModel.pkl', 'rb'))
gold_model = pickle.load(open('Models/GoldPricePrediction.pkl', 'rb'))
car_model = pickle.load(open('Models/CarPricePrediction.pkl', 'rb'))
house_model = pickle.load(open('Models/HousePricePrediction.pkl', 'rb'))
insurance_model = pickle.load(open('Models/insuranceModel.pkl', 'rb'))
loan_model = pickle.load(open('Models/LoanStatusPrediction.pkl', 'rb'))

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/car", methods=['GET','POST'])
def car():
    return render_template('car_form.html')

@app.route("/loan", methods=['GET','POST'])
def loan():
    return render_template('loan_form.html')

@app.route("/bitcoin", methods=['GET','POST'])
def bitcoin():
    return render_template('bitcoin_form.html')

@app.route("/gold", methods=['GET','POST'])
def gold():
    return render_template('gold_form.html')

@app.route("/house", methods=['GET','POST'])
def house():
    return render_template('house_form.html')

@app.route("/insurance", methods=['GET','POST'])
def insurance():
    return render_template('insurance_form.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        if(len([float(x) for x in request.form.values()])==4):
            spx = float(request.form['spx'])
            uso = float(request.form['uso'])
            slv = float(request.form['slv'])
            eurusd = float(request.form['eurusd'])
            
            
            data = np.array([[spx,uso,slv,eurusd]])
            my_prediction = gold_model.predict(data)
            
            return render_template('result.html', prediction = my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==5):
            presentprice = float(request.form['presentprice'])
            fuel = float(request.form['fuel'])
            seller = float(request.form['seller'])
            transmission = float(request.form['transmission'])
            carage = float(request.form['carage'])
            
            
            data = np.array([[presentprice,fuel,seller,transmission,carage]])
            my_prediction = car_model.predict(data)
            
            return render_template('result.html', prediction = my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==6):
            age = int(request.form['age'])
            bmi	= int(request.form['bmi'])
            children = int(request.form['children'])
            smoker = int(request.form['smoker'])
            sex	= int(request.form['sex'])
            region = int(request.form['region'])
            
            data = np.array([[age,bmi,children,smoker,sex,region]])
            my_prediction = insurance_model.predict(data)
            
            return render_template('result.html', prediction=my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==7):
            area = float(request.form['area'])
            availability = float(request.form['availability'])
            location = float(request.form['location'])
            size = float(request.form['size'])
            sqft = float(request.form['sqft'])
            bath = float(request.form['bath'])
            balcony = float(request.form['balcony'])
            
            data = np.array([[area,availability,location,size,sqft,bath,balcony]])
            my_prediction = house_model.predict(data)
            
            return render_template('result.html', prediction = my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==11):
            gender = float(request.form['gender'])
            married = float(request.form['married'])
            dependents = float(request.form['dependents'])
            education = float(request.form['education'])
            selfemployed = float(request.form['selfemployed'])
            appincome = float(request.form['appincome'])
            coappincome = float(request.form['coappincome'])
            amount = float(request.form['amount'])
            amountterm = float(request.form['amountterm'])
            credhistory = float(request.form['credhistory'])
            propertyarea = float(request.form['propertyarea'])
            
            data = np.array([[gender,married,dependents,education,selfemployed,appincome,coappincome,amount,amountterm,credhistory,propertyarea]])
            my_prediction = loan_model.predict(data)
            
            return render_template('loan_result.html', prediction = my_prediction)
        
        else:
            return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)