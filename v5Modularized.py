import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    data = data[[0, 1]]
    data.columns = ['label', 'text']
    data['label'] = np.where(data['label'] == 'spam', 1, 0)
    return data

def preprocess_data(texts):
    vectorizer = CountVectorizer()
    text_features = vectorizer.fit_transform(texts)
    return vectorizer, text_features

def train_model(X, y):
    clf = MultinomialNB()
    clf.fit(X, y)
    return clf

def evaluate_model(clf, X_test, y_test):
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    conf_matrix = confusion_matrix(y_test, pred)
    return accuracy, conf_matrix

def plot_data_distribution(data):
    data['label'].value_counts().plot(kind='bar', title='Distribution of Spam and Non-Spam Messages')
    plt.show()

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['not spam', 'spam'])
    ax.set_yticklabels(['not spam', 'spam'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
    plt.show()

class EmailClassifierUI:
    def __init__(self, master, clf, vectorizer):
        self.master = master
        master.title('Email Classifier')
        master.geometry('500x400')
        self.vectorizer = vectorizer
        self.clf = clf

        self.text_label = tk.Label(master, text='Enter the email text to classify:', font=('Arial', 14))
        self.text_label.pack(pady=10)

        self.text_input = tk.Text(master, height=10, width=50, font=('Arial', 12))
        self.text_input.pack(pady=10)

        self.classify_button = tk.Button(master, text='Classify', font=('Arial', 12), command=self.classify_text, bg='green', fg='white')
        self.classify_button.pack(pady=10)

        self.result_label = tk.Label(master, text='', font=('Arial', 14))
        self.result_label.pack(pady=10)

    def classify_text(self):
        email_text = self.text_input.get('1.0', 'end-1c')
        email_features = self.vectorizer.transform([email_text])
        is_spam = self.clf.predict(email_features)[0]
        if is_spam:
            self.result_label.config(text='The email is spam.', fg='red')
        else:
            self.result_label.config(text='The email is not spam.', fg='green')

if __name__ == '__main__':

    file_path = 'smsspamcollection/SMSSpamCollection'
    data = read_data(file_path)


    vectorizer, text_features = preprocess_data(data['text'])

    X_train, X_test, y_train, y_test = train_test_split(text_features, data['label'], test_size=0.3, random_state=42000)
    clf = train_model(X_train,y_train)

    accuracy, conf_matr = evaluate_model(clf,X_test,y_test)
    print('\nAccuracy on testing set:', round(accuracy*100,3),'%\n')

    plot_data_distribution(data)

    plot_confusion_matrix(conf_matr)
    root = tk.Tk()
    email_classifier_ui = EmailClassifierUI(root,clf,vectorizer)
    root.mainloop()