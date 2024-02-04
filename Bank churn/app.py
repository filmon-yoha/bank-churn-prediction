from flask import Flask,request,jsonify
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
features = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
       'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

@app.route('/',methods=['POST','GET'])
def home():
    try:
        data = request.get_json()
        data = pd.DataFrame([data['features']])
        # Encoding ordinal categorical columns
        ordinal_categorical_columns = ['Education_Level','Income_Category','Card_Category']
        def encoding_education_level(education_level):
            if education_level == 'Uneducated':
                return 0
            elif education_level == 'Unknown':
                return 1
            elif education_level == 'High School':
                return 2 
            elif education_level == 'College':
                return 3
            elif education_level == 'Graduate':
                return 4
            elif education_level == 'Post-Graduate':
                return 5
            elif education_level == 'Doctorate':
                return 6
        def encoding_income_category(income_category):
            if income_category == 'Unknown':
                return 0
            elif income_category == 'Less than $40K':
                return 1 
            elif income_category == '$40K - $60K':
                return 2
            elif income_category == '$60K - $80K':
                return 3
            elif income_category == '$80K - $120K':
                return 4
            elif income_category == '$120K +':
                return 5 
        def encoding_card_category(card_category):
            if card_category == 'Blue':
                return 0
            elif card_category == 'Silver':
                return 1
            elif card_category == 'Gold':
                return 2
            elif card_category == 'Platinum':
                return 3
        def encoding_marital_status(marital_status):
            if marital_status == 'Married':
                return 1
            elif marital_status == 'Single':
                return 2
            elif marital_status == 'Unknown':
                return 3
            elif marital_status == 'Divorced':
                return 0
        data['Education_Level'] = data['Education_Level'].apply(encoding_education_level)
        data['Income_Category'] = data['Income_Category'].apply(encoding_income_category)
        data['Card_Category'] = data['Card_Category'].apply(encoding_card_category)
        data['Marital_Status'] = data['Marital_Status'].apply(encoding_marital_status)
        data['Gender'] = data['Gender'].apply(lambda x:1 if x=='M' else 0)
        prediction = model.predict(data[features])
        prediction_int = int(prediction[0])
        if prediction_int == 1:
            final_prediction = 'Attrited customer'
        elif prediction_int == 0:
            final_prediction = 'Existing customer'
        response = {'prediction': final_prediction}
        return jsonify(response)
    
    except Exception as e:
        error_message = str(e)
        return jsonify({'error': 'Bad Request', 'message': error_message}), 400

if __name__ == '__main__':
    app.run(debug=True)