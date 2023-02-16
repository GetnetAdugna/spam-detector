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

# read the dataset
data = pd.read_csv('smsspamcollection/SMSSpamCollection.csv', sep='\t', header=None)
data = data[[0, 1]]
data.columns = ['label', 'text']

# create a binary label column for spam (1) and not spam (0)
data['label'] = np.where(data['label'] == 'spam', 1, 0)

# preprocess the text data
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(data['text'])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_features, data['label'], test_size=0.3, random_state=42000)
# print((X_train.shape),(X_test.shape))

# train the classifier on the training set
clf = MultinomialNB()
clf.fit(X_train, y_train)

# predict the labels of the testing set
pred = clf.predict(X_test)

# calculate the accuracy and confusion matrix of the model on the testing set
accuracy = accuracy_score(y_test, pred)
conf_matrix = confusion_matrix(y_test, pred)
print('Accuracy on testing set:', accuracy)

# create a bar chart to display the number of spam and non-spam messages in the dataset
data['label'].value_counts().plot(kind='bar', title='Distribution of Spam and Non-Spam Messages')
plt.show()


# display the confusion matrix as a heatmap
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


# create the UI
class EmailClassifierUI:
    def __init__(self, master):
        self.master = master
        master.title('Email Classifier')
        master.geometry('500x400')

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

        # preprocess the email text
        email_features = vectorizer.transform([email_text])

        # classify the email as spam or not spam
        is_spam = clf.predict(email_features)[0]

        # display the result on the UI
        if is_spam:
            self.result_label.config(text='The email is spam.', fg='red')
        else:
            self.result_label.config(text='The email is not spam.', fg='green')


root = tk.Tk()
email_classifier_ui = EmailClassifierUI(root)
root.mainloop()
