import pickle
import os
import re 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import streamlit as st

# ... rest of your code ...
st.write("# Text Emotions Prediction")
t1 = st.text_input("Enter any text>>: ")

def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

file = 'text.txt'
data = read_data(file)
print("Number of instances: {}".format(len(data)))

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

print(create_feature("I love you!"))
print(create_feature(" aly wins the gold!!!"))
print(create_feature(" aly wins the gold!!!!!", (1, 2)))

def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

print("features example: ")
print(X_all[0])
print("Label example:")
print(y_all[0])

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)



# Classifiers 
lsvc = LinearSVC(random_state=123)

# Check if the model is already trained and saved
if os.path.exists('lsvc_model.pkl'):
    # Load the trained model
    with open('lsvc_model.pkl', 'rb') as f:
        lsvc = pickle.load(f)
else:
    # Train the model
    train_acc, test_acc = train_test(lsvc, X_train, X_test, y_train, y_test)
    print("Training Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)
    # Save the trained model
    with open('lsvc_model.pkl', 'wb') as f:
        pickle.dump(lsvc, f)

# ... rest of your code ...
# print the labels and their counts in sorted order 
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1


for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}

# texts = [t1]
# for text in texts: 
#     features = create_feature(text, nrange=(1, 4))
#     features = vectorizer.transform(features)
#     prediction = clf.predict(features)[0]
#     st.write(emoji_dict[prediction])



texts = [t1]
for text in texts: 
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = lsvc.predict(features)[0]
    st.write(emoji_dict[prediction])

