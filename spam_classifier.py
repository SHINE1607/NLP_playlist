import pandas as pd
import nltk 

df_messages = pd.read_csv("./dataset/spam.csv", encoding  = "latin-1")

df_messages = df_messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_messages = df_messages.rename(columns={"v1":"label", "v2":"messages"})
df_temp = df_messages.copy()

# Datacleaning and pre processing
import re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(df_messages)):
    review = re.sub('[^a-zA-Z]', ' ', df_messages['messages'][i])
    review = review.lower()
    review = review.split()
    
    review = [wordnet.lemmatize(word) for word in review if word not in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)
    
    
    
#Cretaing bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()     


# #creating idf-idf model
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv = TfidfVectorizer()
# X = cv.fit_transform(corpus).toarray()

#creating the target array
y = pd.get_dummies(df_messages["label"], drop_first = True)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.20, random_state = 42)
 
#Training and fittimng the model 
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

#computing confusion amtrix and accuray score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = confusion_matrix(y_test,y_pred, labels=[1,0])
tp, fn, fp, tn = confusion_matrix(y_test,y_pred,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

accuracy_score = (tp + tn)/len(y_test)
