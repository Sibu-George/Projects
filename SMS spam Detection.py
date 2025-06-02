import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix


d_in=pd.read_csv(r"C:\Users\Metatron\Downloads\archive (4)\spam.csv",encoding='latin-1')

d_in.rename(columns= {'v1':'spam_ham','v2':'massage'}, inplace = True)

d_in['isSpam']=[0 if x=='ham' else 1 for x in d_in['spam_ham']]

d_in=d_in.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1) 
d_in.groupby('spam_ham').describe()

def spam_ham():
    fig=sns.countplot(x=d_in['spam_ham'])
    fig.set_title("Number of Spam and Ham")
    fig.set_xlabel("Classes")
    fig.set_ylabel("Number of Data points")
    plt.show()

x_train,x_test,y_train,y_test=train_test_split(d_in['massage'],d_in['isSpam'])
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

full_dataset = pd.concat([x_train, x_test], axis=0)
full_labels = pd.concat([y_train, y_test], axis=0)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

def complete_75():
    sns.countplot(x=full_labels, ax=axes[0])
    axes[0].set_title("Complete Dataset")
    axes[0].set_xlabel("Classes")
    axes[0].set_ylabel("Number of Data points")

    sns.countplot(x=y_train, ax=axes[1])
    axes[1].set_title("Training Set 4179 records(75%)")
    axes[1].set_xlabel("Classes")
    axes[1].set_ylabel("Number of Data points")
    plt.show()

cv = CountVectorizer()
word_count = cv.fit_transform(x_train)
word_count = word_count.toarray()
x=pd.DataFrame(word_count, columns=cv.get_feature_names_out()).head()


model=MultinomialNB()
model.fit(word_count, y_train)

test=['''Subject: Urgent: Verify Your Account Information Now
Dear User,
Your account has been hacked, and we need you to verify your account information immediately. 
Click the link below to log in and secure your account:
[Phishing Link]
Sincerely,
Supposedly Bank
''']

t_count=cv.transform(test)
model.predict(t_count)

x_test_count=cv. transform(x_test)
print(classification_report(y_test, model.predict(x_test_count)))
y_pred = model.predict(x_test_count)

cm = confusion_matrix(y_test, y_pred)

def confusion():
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print('The accuracy of Naive Bayes Model is :',model.score(x_test_count,y_test))



def menu():
    while True:
        print("\nSpam Classification Model Menu:")
        print("1. Show Spam VS Ham")
        print("2. Complete Dataset and 75% dataset")
        print("3. Show Confusion Matrix")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            spam_ham()
        elif choice == '2':
            complete_75()
        elif choice == '3':
            confusion()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

menu()
