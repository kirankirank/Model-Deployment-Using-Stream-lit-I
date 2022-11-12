# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:25:49 2022

@author: kiran
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
#Multi_linear= pickle.load(open("mlr.pkl", "rb"))
od=pickle.load(open('labelencoder.pkl', 'rb'))
model=pickle.load(open('mlr.pkl', 'rb'))
#yp = model.predict(od.transform([['hybrid',60,60,120,33]]))
#yp
# HtmlFile = open("C:/Users/kiran/Streamlite/templates/index.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code,width=700, height=1000)
def prediction(engine, hp, vol, sp,wt):  
    b = int(hp)
    c= int(vol)
    d = int(sp)
    e = int(wt)
    prediction = model.predict([[od.transform([engine]),  b,c,d,e]])
    print(prediction)
    return prediction
def main():
      # giving the webpage a title
    st.title("Car fuel Efficiency")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
     
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    engine = st.text_input("Engine Type(Please menction Petrol or Hybrid or Diesel or Lpg or Cng)", "  ")
    hp = st.text_input("HP(Range 49-322)", " ")
    vol = st.text_input("VOL(Range 50-160)", " ")
    sp = st.text_input("SP(Range 16-52)", " ")
    wt = st.text_input("WT(Range 16-52)", " ")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(engine,  hp, vol, sp,wt )
    st.success('The output is : {}'.format(result[0]))
     
if __name__=='__main__':
    main()