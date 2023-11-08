# Core Pkgs
import streamlit as st 
import altair as alt
from PIL import Image
import plotly.express as px 
import streamlit_authenticator as stauth
# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import string
import math
# Utils
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier.pkl","rb"))
pipe_s=joblib.load(open("models/sentiment_classifier.pkl","rb"))

import yaml

# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table
names = ['Omar Ben Rhaiem']
usernames = ['jsmith','rbriggs']
passwords = ['123456']
hashed_passwords = stauth.Hasher(passwords).generate()
with open(os.path.join(os.path.expanduser('~'),'tourisfair sentiment analyzer\App\config.yaml')) as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials']['names'],
    config['credentials']['usernames'],
    config['credentials']['passwords'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'])
name, authentication_status, username = authenticator.login('Login', 'main')
from load_css import local_css

local_css(os.path.join(os.path.expanduser('~'),'tourisfair sentiment analyzer\App\styles.css'))
import spacy 
from spacytextblob.spacytextblob import SpacyTextBlob
def get_keywords(text): #this function extract keywords based on their score
    nlp = spacy.load('en_core_web_sm')
    spacy_text_blob = SpacyTextBlob()
    nlp.add_pipe(spacy_text_blob)
    doc = nlp(text)
    result=doc._.sentiment.assessments
    neg_keywords=[kw[0] for kw in result if (kw[1]<-0.2)]
    neu_keywords=[kw[0] for kw in result if ( -0.2<=kw[1]<=0.2)]
    pos_keywords=[kw[0]for kw in result if (kw[1]>0.2)]
    return neg_keywords,pos_keywords,neu_keywords

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]
def predict_sentiment(docx):
    results = pipe_s.predict([docx])
    return results[0]
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
def get_prediction_proba_s(docx):
    results = pipe_s.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
    st.title("Tourisfair Sentiment and Emotions Analyzer")
    menu = ["Emotion","Sentiment","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Emotion":
        add_page_visited_details("Home",datetime.now())
        st.subheader("Emotional Analyzer")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2  = st.beta_columns(2)

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            
            add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                st.write("Keywords:{}".format(get_keywords(raw_text)))
                #t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

                #st.markdown(t, unsafe_allow_html=True)
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)



    elif choice == "Sentiment":
        add_page_visited_details("Home",datetime.now())
        st.subheader("Sentiment Analyzer")

        with st.form(key='sentiment_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2  = st.beta_columns(2)

            # Apply Fxn Here
            prediction = predict_sentiment(raw_text)
            probability = get_prediction_proba_s(raw_text)
            
            add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                #emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}".format(prediction))
                st.write("Confidence:{}".format(np.max(probability)))
                #st.write("Keywords:{}".format(get_keywords(raw_text)))
                #t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

                #st.markdown(t, unsafe_allow_html=True)
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_s.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["sentiment","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiment',y='probability',color='sentiment')
                st.altair_chart(fig,use_container_width=True)	



    else:
        st.subheader("About the application")
        add_page_visited_details("About",datetime.now())
        st.write("Tourisfair Sentiment and emotional analyzer is a web app developped to extract emotions from users's reviews about touristic activities")
        st.subheader("About the Developper")
        st.write("The web application was developped by Omar Ben Rhaiem as a graduation project.")
        st.subheader("Contact Me")
        st.write(" [Facebook](https://www.facebook.com/omar.xbenrhaiem/)")
        st.write(" [LinkedIn](https://www.linkedin.com/in/omar-ben-rhaiem-bab84b1a2/)")
        
        





if __name__ == '__main__':
        if authentication_status:
                authenticator.logout('Logout', 'main')
                main()
        elif authentication_status == False:
                st.error("Username/password is incorrect")
        elif authentication_status == None:
                st.warning("Please enter your username and password")
