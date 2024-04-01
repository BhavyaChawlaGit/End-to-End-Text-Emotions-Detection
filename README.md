# Text Emotion Detection (End to End)
This project uses a LinearSVC model to predict the emotion of a given text.  

## Dataset
The dataset used for this project is a text file named text.txt, which can be found [here](https://github.com/amankharwal/Website-data/blob/master/text.txt).  
**Number of instances: 7480**  
The file contains labeled text data, where each line consists of a label and a text string.   
The labels represent different emotions, and the text strings are sentences or phrases that express these emotions.  

The data is preprocessed by converting the labels to integers and creating feature vectors for the text strings.   
The feature vectors are created using the create_feature function, which extracts n-grams from the text strings.  

Here's a snippet of the code that preprocesses the data:  
```
X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))
```



## Requirements
- Python 3.7 or higher
- Libraries: sklearn, pickle, streamlit
You can install the required libraries using pip:
```
pip install -r requirements.txt
```

## How to Run

- Clone this repository to your local machine.
-  Navigate to the project directory in your terminal.
-  Run the following command to start the Streamlit server and open the application in your browser:

```
streamlit run train_model.py
```

The first time you run this script, it will train the model and save it to a file named lsvc_model.pkl.   
For subsequent runs, it will load the model from this file instead of training it again.  

You can interact with the application in your browser. Any changes you make to the script while the server is running will automatically be reflected in the application.  

## Model Training
The model is trained using the train_test function, which trains the model and returns the training and test accuracies.  


```
# Train the model
train_acc, test_acc = train_test(lsvc, X_train, X_test, y_train, y_test)
print("Training Accuracy: ", train_acc)
print("Test Accuracy: ", test_acc)
```


### Label Frequencies

The script also prints the frequencies of each label in the data:  


```
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))
```

## Model Selection

The [textemotion.py](https://github.com/BhavyaChawlaGit/End-to-End-Text-Emotions-Detection/blob/f0a53d49c26640b973944a84297afebac597a68d/textemotion.py) script compares the performance of four different models: 
- SVC
- LinearSVC
- RandomForestClassifier
- DecisionTreeClassifier

It trains each model on the training data and calculates the training and test accuracies.  
The script then selects the model with the highest test accuracy for further tuning. In this case, the LinearSVC model was selected because it had the highest test accuracy.  

Here's the code that trains the models and calculates their accuracies:  

```
models = [svc, lsvc, rforest, dtree]
for model in models:
    train_acc, test_acc = train_test(model, X_train, X_test, y_train, y_test)
    print(f"Model: {model.__class__.__name__}, Training Accuracy: {train_acc}, Test Accuracy: {test_acc}")
```


## Running textemotion.py
To run textemotion.py, navigate to the project directory in your terminal and run the following command:  
```
python textemotion.py
```
This script reads data from text.txt, compares the performance of four different models, and prints the number of instances in the data.  

---

## Scope for Improvement
While the current implementation of the text emotion prediction model provides a good starting point, there are several areas where the model and results needs to be improved  
The current model appears to be overfitting, as it performs well on the training data but poorly on the test data.  

```
| Classifier                | Training Accuracy | Test Accuracy |
| ------------------------- | ----------------- | ------------- |
| SVC                       |         0.1458890 |     0.1410428 |
| LinearSVC                 |         0.9923128 |     0.5802139 |
| RandomForestClassifier    |         0.9911430 |     0.4304813 |
| DecisionTreeClassifier    |         0.9988302 |     0.4632353 |
```

**Best classifier:  LinearSVC**

---




