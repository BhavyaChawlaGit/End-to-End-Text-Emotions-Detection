import streamlit as st
from joblib import load
import re
from collections import Counter

@st.cache(allow_output_mutation=True)
def load_resources():
    clf = load('SVC_model.joblib')  # Or whichever model you choose
    vectorizer = load('vectorizer.joblib')
    return clf, vectorizer

@st.cache
def create_feature(text, nrange=(1, 4)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

clf, vectorizer = load_resources()

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}

def predict_emotion(text):
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform([features])  # Note the list around 'features' for correct shape
    prediction = clf.predict(features)[0]
    return prediction

# Streamlit UI
st.write("# Text Emotion Prediction")
t1 = st.text_input("Enter any text to predict its emotion:", "")

if t1:  # Only predict if there is some input
    prediction = predict_emotion(t1)
    emoji = emoji_dict.get(prediction, "😶")  # Fallback emoji if prediction not in dictionary
    st.write(f"Predicted Emotion: {emoji}")