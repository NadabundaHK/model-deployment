import streamlit as st
import numpy as np
import joblib
import os

attribute_info = """
                 - Profession: Healthcare, Engineer, Lawyer, Artist, Doctor, Homemaker, Entertainment, Marketing, Executive
                 - Work_Experience: 0-15
                 - Spending_Score: Low, High, Average
                 - Gender: Male and Female
                 - Family_Size: 1-10
                 - Var_1: Cat_1, Cat_2, Cat_3, Cat_4, Cat_5, Cat_6, Cat_7
                 - Age: 15-100
                 - Ever_Married: 1.Yes, 0.No
                 - Graduated: 1. Yes, 0. No
                 """

# Dictionaries for encoding categorical data
prof = {'Healthcare': 1, 'Engineer': 2, 'Lawyer': 3, 'Artist': 4, 'Doctor': 5,
        'Homemaker': 6, 'Entertainment': 7, 'Marketing': 8, 'Executive': 9}
spen = {'Low': 1, 'High': 2, 'Average': 3}
gen = {'M': 1, 'F': 2}
var = {'Cat_1': 1, 'Cat_2': 2, 'Cat_3': 3, 'Cat_4': 4, 'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7}
mar = {'No': 1, 'Yes': 2}
grad = {'Yes': 1, 'No': 2}

def get_value(val, my_dict):
    """Utility function to get the value from the dictionary based on the key"""
    for key, value in my_dict.items():
        if val == key:
            return value

def load_model(model_file):
    """Load the trained model from a file"""
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    """Main function to run the Streamlit ML app"""
    st.subheader("ML Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    # Input form for user data
    st.subheader("Input Your Data")
    Profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 
                                            'Doctor', 'Homemaker', 'Entertainment', 'Marketing', 'Executive'])
    Experience = st.number_input('Work Experience', 1, 10)
    Spending_score = st.selectbox('Spending score', ['Low', 'High', 'Average'])
    Gender = st.radio('Gender', ['M', 'F'])
    Family_Members = st.number_input('Family size', 1, 10)
    Var_1 = st.selectbox('Var 1', ['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'])
    Age = st.number_input('Age', 18, 90)
    Married = st.selectbox('Ever Married', ['No', 'Yes'])
    Graduated = st.selectbox('Graduated', ['Yes', 'No'])

    with st.expander("Your Selected Options"):
        result = {
            "Profession": Profession,
            "Work Experience": Experience,
            "Spending Score": Spending_score,
            "Gender": Gender,
            "Family Size": Family_Members,
            "Var 1": Var_1,
            "Age": Age,
            "Ever Married": Married,
            "Graduated": Graduated,
        }
        st.write(result)

    # Encode the selected data
    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in prof:
            res = get_value(i, prof)
            encoded_result.append(res)
        elif i in spen:
            res = get_value(i, spen)
            encoded_result.append(res)
        elif i in gen:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in var:
            res = get_value(i, var)
            encoded_result.append(res)
        elif i in mar:
            res = get_value(i, mar)
            encoded_result.append(res)
        elif i in grad:
            res = get_value(i, grad)
            encoded_result.append(res)


    st.subheader('Prediction Result')

    # Add missing features if needed to make the total features 23
    missing_features_count = 22 - len(encoded_result)
    encoded_result.extend([0] * missing_features_count)  # Add default values for missing features


    single_array = np.array(encoded_result).reshape(1, -1)

    try:
        # Load the trained model
        model = load_model("logistic_regression_model (1).pkl")

        # Make a prediction
        prediction = model.predict(single_array)
        pred_proba = model.predict_proba(single_array)

        # Probability scores for each segment
        pred_probability_score = {
            'A': round(pred_proba[0][0] * 100, 4),
            'B': round(pred_proba[0][1] * 100, 4),
            'C': round(pred_proba[0][2] * 100, 4),
            'D': round(pred_proba[0][3] * 100, 4),
        }

        # Display result based on prediction
        if prediction == 0:
            st.info("You are identified as part of Segment A.")
        elif prediction == 1:
            st.info("You are identified as part of Segment B.")
        elif prediction == 2:
            st.warning("You are identified as part of Segment C.")
        elif prediction == 3:
            st.error("You are identified as part of Segment D.")

        # Display the probability scores
        st.write("Probability scores for each segment:")
        st.write(pred_probability_score)

    except Exception as e:
        st.error(f"An error occurred: {e}")

