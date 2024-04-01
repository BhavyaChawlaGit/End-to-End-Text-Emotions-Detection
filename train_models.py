import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from joblib import dump

def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            label, text = line.split(']', 1)
            label = ' '.join(label[1:].strip().split())
            text = text.strip()
            data.append([label, text])
    return data

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

# Assuming your data file is named text.txt and properly formatted
file = 'text.txt'
data = read_data(file)
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all, y_all = [], []
for label, text in data:
    y_all.append(label)  # Assuming the labels are already in the correct format; adjust if necessary
    X_all.append(create_feature(text, nrange=(1, 4)))

vectorizer = DictVectorizer(sparse=True)
X_all = vectorizer.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

# Train classifiers
classifiers = {
    "SVC": SVC(),
    "LinearSVC": LinearSVC(random_state=123),
    "RandomForest": RandomForestClassifier(random_state=123),
    "DecisionTree": DecisionTreeClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    # You can add code here to print out accuracy or other metrics if you'd like
    dump(clf, f'{name}_model.joblib')

# Save the vectorizer too
dump(vectorizer, 'vectorizer.joblib')