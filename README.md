# E-mail-Spam-Detection

The major issues faced by all the email users are spam mails which contain unwanted
information and data and some fake data to spoil the life of the people and also some mails
which cause harmful effects. To reduce this risk and to save the people from this danger of spam
mails, we are building this Gmail Spam Detection Model
---------Tools and Libraries-------------
As the development environment during the project Spyder, Python is used as the software
language. In the project, i used NumPy, Pandas, Scikit-learn libraries, Seaborn,NLTK,re
library,string .
• NumPy is a library for the Python programming language that supports large,
multidimensional arrays and matrices and adds high-level mathematical functions for
working on these arrays.
• Pandas simplifies data analysis and preprocessing. It was used in the project to receive,
process, and analyze the data kept in the csv file.
• Scikit-learn is a software machine learning library for the Python programming language.
It features various classification, regression and clustering algorithms including Logistic
Regression, Vectorizer, Accuracy and split functions. In the next steps, i will explain in
detail which features of the Scikit-learn library we benefit from.
• I used the Seaborn library for the Data Visualization.
• NLTK, NLTK (Natural Language Toolkit) is a library for the Python programming
language in the field of natural language processing (NLP). NLTK can be used for
language analysis, text classification, document sorting, information extraction, semantic
analysis and many more NLP tasks. In the project, removing stopwords, stemming,
lemmatization, tokenizar operations are done using NLTK library.
• The "re" library is part of Python's standard library and is used as a re module. It provides
various functions and methods for creating and using regular expressions. These functions
include search, match, shred, replace and more. I used the re library to remove emojis and
numeric values.
• In Python, the "string" module is a standard library that provides some helper functions
and constants for text manipulation and manipulation. I used the String library to convert
text to lowercase, removing punctuation and underline spacing.

----Dataset----------------
I used E-mail Spam Detection Dataset from kaggle.com. The data contains one message per
line. Each row consists of two columns: contains the label (raw or spam) and the text column
contains the raw text. There are 5171 texts in total 3672 ham, 1499 spam mail.

----Result---------------
According to the classification report, the F1-measure highest algorithm for this project is
Support vector machine. Thus, the best classification algorithm is SVM. 
