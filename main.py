import streamlit as st 
import tensorflow as tf
import numpy as np
from tensorflow import keras


def mode_prdiction(test_img):
    model=tf.keras.models.load_model("trained_model (1).h5")
    image=tf.keras.preprocessing.image.load_img(test_img,target_size=(64,64))
    input_arr=keras.preprocessing.image.img_toarray(image)
    input_arr=np.array([input_arr])# Conver single
    predictions=model.predict(input_arr)
    return np.argmax(predictions) 


#side bar
st.sidebar.title("DashBord")
app_mode=st.sidebar.selectbox("Select Box ",["Home","About Project","Prediction"])

#Main page
if(app_mode=="Home"):
    
    image_pth="vegi.jpg"
    st.image(image_pth)
    st.header("Fruit and Vegitable  Recognition System")
#About Page    
elif(app_mode=="About Project"):
    st.header("About Project") 
    st.subheader("About Data Set")
    st.text("This dataset contains images of the following food items:") 
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("content")
    st.text("This dataset contains three folders:")
    st.text("1.train (100 images each)")
    st.text("2.test (10 images each)")
    st.text("3.validation (10 images each)")
    st.subheader("Group Members")
    st.text("1.Theminda Kirulapana")
    st.text("2.Haritha Migara") 
    st.text("3.Nadun sudeep")
#Prediction page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")  
    test_img=st.file_uploader("Choose an Image")
    if(st.button("SHOW IMAGE")):
        st.image(test_img,width=2,use_column_width=True)
    #prection Button
    if(st.button("PREDICT OUR MODEL")):
        st.write("Our prediction Apple")    
    

