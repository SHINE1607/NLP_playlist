import nltk 

#stemming
from nltk.stem  import PorterStemmer
from nltk.corpus import stopwords

#lemmatizing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#bag of words 



paragraph = """“In life, we’ve seen, within this part of the 
world, great people, like Al-Khawarizmi who invented algorithm. 
Globally, Newton. Henry Ford, the Wright Brothers, 
Albert Einstein and Elon Musk. See, you are in rush, 
You want to go to places that nobody has been. 
You are re-inventing a certain industry, from the rocket 
industry with SpaceX to the car industry with Tesla. 
What’s your life mission? Why do you do whatever you do?
Sure, first of all, thank you for having me. 
It’s an honor to be here. And I’m having a really great 
time with my kids in Dubai. It’s really been fantastic. 
I really encourage anyone who hasn’t been to visit, what a 
great city! Thank you.

And, in terms of the motivations, I used to like this sort of… 
kind of a long version of the explanation but… essentially, 
when I was a kid I was wondering, what’s the meaning of life? 
Like, why are we here? What is it all about? And I came to the 
conclusion that what really matters is trying to understand the 
right questions to ask. And the more that we can increase the 
scope and scale of human consciousness, the better we’re able 
to ask these questions. And… So, I think there are certain 
things that are necessary to ensure that the future is good. 
And… some of those things are in the long term having long term 
sustainable transport and sustainable energy generation. 
And to be a space exploring civilization. And for humanity to 
be out there among the stars. And be a multi-planetary species. 
I think that being a multi-planetary species and being out there 
among the stars is important for the long-term survival of humanity. 
And, that’s one reason, kind of like life insurance for life 
collectively. Life as we know it. But then the part that I find 
personally most motivating is that it creates a sense of adventure, 
and it makes people excited about the future. If you consider two 
futures, one where we are forever confined to Earth until eventually 
something terrible happens. Or another future where we are out there 
on many planets, maybe even going beyond the solar system. I think that space 
invasion is incredibly exciting and inspiring. And there need to 
be reasons to get up in the morning. You know, life can’t just 
be about solving problems. Otherwise, what’s the point? There’s got to be 
things that people find inspiring and make life worth living."""

#tokenizing the sentneces 
sentences = nltk.sent_tokenize(paragraph)

############# tokenizing words #############
words = nltk.word_tokenize(paragraph)

############# stemming #############
stemmer = PorterStemmer()
# stop words doesnt add values to sentiments of the sentences
# stemming
problem with stemming : most of the stemmed words doesnt make sense
sentences_stemmed, sentences_lemmatized = [""] * len(sentences), [""] *  len(sentences)
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i]), 
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words("english"))]
    
    sentences_stemmed[i] = " ".join(words)

############# lemmatizing #############
lemmatizer = WordNetLemmatizer()

#lammatizing 
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences_lemmatized[i] = ' '.join(words)      
    
############# Bag of words #############

#cleaning of text 
import re #re = > regular expression
corpus = []     #for storingsentnces aftre cleaning 
for i in range(len(sentences)):
    review = re.sub("[a-z^A-Z]", " ", sentences[i])   #removing special character from the sentences
    review = review.lower()     #lowering all the characters
    review = review.split()  #sentence => list of words 
    review  =  [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words("english"))]
    corpus.append(review)   #cleaned sentences as list of words 
    
#creating bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
    
    
















