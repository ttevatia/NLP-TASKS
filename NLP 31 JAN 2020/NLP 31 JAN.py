#!/usr/bin/env python
# coding: utf-8

# # TASK-6 17BCE2029(TAVISH)

# ## A. TYPES OF STEMMERS

# ### I. REGEX STEMMER 

# In[1]:


import nltk
from nltk.stem import RegexpStemmer
stemmerregexp=RegexpStemmer('ing')
stemmerregexp.stem('running')


# ### II. SNOWBALL STEMMER

# In[7]:


import nltk 
from nltk.stem import SnowballStemmer
SnowballStemmer.languages
frstemmer = SnowballStemmer('french')
frstemmer.stem('manges')


# ### III. LANCASTER STEMMER

# In[8]:


import nltk
from nltk.stem import LancasterStemmer
lancaster=LancasterStemmer()
lancaster.stem('running')


# ### IV. PORTEWR STEMMER 

# In[10]:


import nltk
from nltk.stem import PorterStemmer
stemmerporter = PorterStemmer()
stemmerporter.stem('running')


# #### "___Regex Stemmer unique utility___"

# In[14]:


#porter
print("For PorterStemmer ",stemmerporter.stem('taker'))
#SnowballStemmer
stt = SnowballStemmer('english')
print("For Snowball ",stt.stem('taker'))
#regex
reg = RegexpStemmer('[$r|$s]')
print("For RegexStemmer" , reg.stem('taker'))
#lancasterStemmer
print("For lancaster ",lancaster.stem('taker'))


print("Thus in above case RegexStemmer performs better than other Stemmers")


# ## B. LEMMATIZER  

# In[17]:


from nltk.stem import WordNetLemmatizer


# In[18]:


lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better",pos='v'))


# In[19]:


print(lemmatizer.lemmatize("rocks",pos='a'))


# In[20]:


print(lemmatizer.lemmatize("unhappy"))


# ## C. COMPARISON OF STEMMER AND LEMMATIZER

# ### i. STEMMER

# In[2]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
example = "Am quick brown fox jumps over a lazy dog"
example = [stemmer.stem(token) for token in example.split(" ")]
print(" ".join(example))


# ### ii. LEMMATIZER 

# In[5]:


from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
example1 = "Am quick brown fox jumps over a lazy dog"
example1 = [stemmer.stem(token) for token in example1.split(" ")]
print(" ".join(example1))


# '''Stemming usually refers to a crude heuristic process that 
# chops off the ends of words in the hope of achieving this goal 
# correctly most of the time, 
# and often includes the removal of derivational affixes. 
# Lemmatization usually refers to doing things properly with
# the use of a vocabulary and morphological analysis of words, 
# normally aiming to remove inflectional endings only and to
# return the base or dictionary form of a word, which is known
# as the lemma'''  

# LEMMATIZER IS OFTEN BETTER THAN STEMMER/
# 

# ## D. COUNT VECTORIZERS

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.','This document is the second document.', 'And this is the third one.',
'Is this the first document?']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())


# CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.

# In[ ]:




