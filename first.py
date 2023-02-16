import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests

# download the dataset from a remote URL
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
# r = requests.get(url)
# with open('smsspamcollection.zip', 'wb') as f:
#     f.write(r.content)

# import zipfile
# with zipfile.ZipFile('smsspamcollection.zip', 'r') as zip_ref:
#     zip_ref.extractall()

data = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', header=None)
data = data[[0, 1]]
data.columns = ['label', 'text']

# create a binary label column for spam (1) and not spam (0)
data['label'] = np.where(data['label'] == 'spam', 1, 0)

# preprocess the text data
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(data['text'])

# train the classifier on the entire dataset
clf = MultinomialNB()
clf.fit(text_features, data['label'])

# create the UI
class EmailClassifierUI:
    def __init__(self, master):
        self.master = master
        master.title('Email Classifier')

        self.text_label = tk.Label(master, text='Enter the email text to classify:')
        self.text_label.pack()

        self.text_input = tk.Text(master, height=10, width=50)
        self.text_input.pack()

        self.classify_button = tk.Button(master, text='Classify', command=self.classify_text)
        self.classify_button.pack()

        self.result_label = tk.Label(master, text='')
        self.result_label.pack()

    def classify_text(self):
        email_text = self.text_input.get('1.0', 'end-1c')

        # preprocess the email text
        email_features = vectorizer.transform([email_text])

        # classify the email as spam or not spam
        is_spam = clf.predict(email_features)[0]

        # display the result on the UI
        if is_spam:
            self.result_label.config(text='The email is spam.')
        else:
            self.result_label.config(text='The email is not spam.')

root = tk.Tk()
email_classifier_ui = EmailClassifierUI(root)
root.mainloop()
