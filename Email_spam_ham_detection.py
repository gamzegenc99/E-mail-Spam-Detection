import numpy as np
import pandas as pd
import string 
import re
import os
import nltk
from typing import List
import warnings
warnings.filterwarnings('ignore')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from collections import Counter #find to commmon words

#-----------------------------Loading Dataset-----------------------------
trainData = pd.read_excel(r"C:\Users\gamze\OneDrive\Masaüstü\Gamze\LESSONS OF FOURTH CLASS\2nd Term\intro to Web Application Security\spam_ham_dataset.xlsx")
trainData['label'] = trainData['label'].map({'spam': 1, 'ham': 0}).astype(int)
#print(trainData.head())
#print(trainData['label'].value_counts()) # 0->   3672, 1 ->   1499

# -------------------------Data Visualization--------------------------------------------------------
trainData["label"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)
plt.title("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()
#---------------------------Word Cloud----------------------------------------------------------------------------
#
wordcloud = WordCloud(width=600, height=400).generate(' '.join(trainData["text"]))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#---------------------------PREPROCESSING---------------------------------------------------------

def tokenizer(text):# Tokenizer is spliting mail text to words
    return text.split()

"--------Removal Emoticons,punctation, numbers, char -------------------------------"
punctation = string.punctuation #punctuation ='''!()-[]{};':'"\,<>./?@#$%^&*_~'''
def dataCleaning(text):
    text =  str(text).translate(str.maketrans("", "",punctation))
    text = re.sub(r"\d+","", str(text)) #remove numeric values    
    text =  str(text).replace("\n", " ") # alt satır boşluklarının kaldırılması
    text = text.lower()
    text = [i for i in str(text).split() if (i.isalpha())]
    return " ".join(text)

def removal_emoticons (text):
    emoticons = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"                                 
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"                                 
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoticons.sub(r"",text)


"Stemming Stemming is the process of removing of suffix to convert the word into core values. For example, converting waits, waiting, waited to the core word wait"
stemmer = SnowballStemmer("english", ignore_stopwords=False)
def stem(text):
    wordlist = nltk.word_tokenize(text)
    stemWords = [stemmer.stem(word) for word in wordlist]
    return " ".join(stemWords)

"""Lemmitaziton
 It is the process of finding lemma of a word depending on their meaning."""
lemma = WordNetLemmatizer()
def lemmit (text):
    return " ".join([lemma.lemmatize(word) for word in str(text).split()])


"""Removal stopwords"""
stopword = set(stopwords.words("english"))
def stop (text):
    return " ".join([word for word in str(text).split() if word not in stopword])


#------ after preprocessing, get the data set----------------------------------------------------------
trainData["text"]= trainData["text"].apply(lambda text : stop(text))
trainData["text"]= trainData["text"].apply(lambda text : tokenizer(text))
trainData["text"]= trainData["text"].apply(lambda text : dataCleaning(text))
trainData["text"]= trainData["text"].astype(str).apply(lambda text : removal_emoticons(text))
trainData["text"]= trainData["text"].apply(lambda text : stem(text))
#print(trainData['text'][3])
trainData["text"]= trainData["text"].apply(lambda text : lemmit(text))
#print(trainData['text'][5])

#pd.concat([pd.concat([trainData["label"],trainData["text"]] , axis=1)]).to_excel('Emaildata_Afterpreprocessing.xlsx')
print(trainData.head())


# ----------- Find most common 50 words--------------------------------------

def find_most_common_words(text_list: List[str], top_n: int):
    all_words = []
    for text in text_list:
        words = str(text).split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    top_words_dict = {word: count for word, count in top_words}
    return top_words_dict
text_list = trainData["text"].tolist()
top_words_dict = find_most_common_words(text_list, top_n=50)
print("Most common words:")
for word, count in top_words_dict.items():
    print(f"{word}: {count}")

#----------------------- Vectorization-------------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(trainData['text'].values.astype('U'))
trainDataX=X.toarray() 
trainDataY = trainData["label"].values
print("number of features" ,trainDataX.shape) 
#number of features (5171, 44578)

"""
Features at raw data 5171 instance ->57832 features then
after all preprocess  5171 instance->  44578 features
"""
#Splitting data into train and test data
xTrain, xTest, yTrain, yTest = train_test_split(trainDataX, trainDataY, random_state = 0, test_size = 0.2, shuffle = False)


#------------------------ModelTraining----------------------------------------------------------

def logistic(xTrain,yTrain):
    lr = LogisticRegression()
    lr.fit(xTrain, yTrain)
    return lr

def randomForest(x_train,y_train):
    rfc = RandomForestClassifier(random_state = 20,n_estimators=100,n_jobs=-1)
    rfc.fit(x_train, y_train)
    return rfc

def decisionTree(x_train,y_train):
    dct = DecisionTreeClassifier(max_depth =5, random_state = 42)# max depth 10 idi 5 yaptım
    dct.fit(x_train, y_train)
    return dct
def svm(x_train, y_train):
    svm = SVC()
    svm.fit(x_train, y_train)
    return svm

def GnaiveBayes(x_train,y_train):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    return gnb

def naiveBayesMultinomial(x_train, y_train):
    nbm = MultinomialNB()
    nbm.fit(x_train, y_train)
    return nbm


#--------------Plotting Decision Tree ------------------------------------------
fig = plt.figure(figsize=(35,30))
tree.plot_tree(decisionTree(xTrain,yTrain), filled=True)
plt.show()
#------------Creating confusing matrix--------------------------------------------
# for logistic regression
y_pred = logistic(xTrain, yTrain).predict(xTest)
con_mat = confusion_matrix(yTest,y_pred)
print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
# for svm
y_pred = svm(xTrain, yTrain).predict(xTest)
con_mat = confusion_matrix(yTest,y_pred)
print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="viridis")


#----------------- Prediction--------------------------------------------------------------

def predictReport(xTrain, xTest,yTrain, yTest):##accuracy report
    print("Classification report for Gaussian Naive Bayes:\n")
    print(classification_report(yTest, GnaiveBayes(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for Random Forest:\n")
    print(classification_report(yTest, randomForest(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for decisionTree:\n")
    print(classification_report(yTest, decisionTree(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for  Logistic :\n")
    print(classification_report(yTest, logistic(xTrain,yTrain).predict(xTest))+"\n")
    print("Classification report for SVM:\n")
    print(classification_report(yTest, svm(xTrain, yTrain).predict(xTest)) + "\n")
    print("Classification report for Naive Bayes Multinomial:\n")
    print(classification_report(yTest, naiveBayesMultinomial(xTrain, yTrain).predict(xTest)) + "\n")
   

print("Report for classification Prediction")
predictReport(xTrain, xTest, yTrain, yTest)

#according to result SVM is the best model

#------------------------------Predited Model---------------------------------------------------------------

    
def prediction(inp,x,y):
    input_mail = [inp]
    transformed_data = tfidf.transform(input_mail)
    transformed_data_dense = transformed_data.toarray() 
    predct = svm(x,y).predict(transformed_data_dense) #burayı linear ile değiştirdim en iyi model o olduğu için 
    
    if predct == 1:
        print("\nSpam mail")
    else:
         print("\nHam mail")
        
inptt= "Your free ringtone is waiting to be collected. Simply text the password \MIX\" to 85069 to verify. Get Usher and Britney. FML"
prediction(inptt, xTrain, yTrain)
