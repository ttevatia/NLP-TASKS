#!/usr/bin/env python
# coding: utf-8

# # TASK - 7
# # CLASSIFICATION 
# ## NAME - TAVISH TEVATIA
# ## REG - 17BCE2029

# In[8]:


def gender_features(word):
    return {'last letter ':word[-1]}


# In[9]:


gender_features('obama')


# In[1]:


from nltk.corpus import names


# In[3]:


names.words()


# In[10]:


print(len(names.words()))


# In[11]:


labeled_names = ([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')])


# In[13]:


import random
random.shuffle(labeled_names)


# In[15]:


featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]


# In[16]:


train_set, test_set = featuresets[5000:], featuresets[:2000]


# In[18]:


import nltk 
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features()


# In[19]:


classifier.classify(gender_features("Ishaan"))


# In[20]:


print(nltk.classify.accuracy(classifier,test_set))


# # PART OF SPEECH TAGGING

# In[26]:


import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 


txt = ("Sukanya, Rajib and Naba are my good friends. " 
"Sukanya is getting married next year. " 
	"Marriage is a big step in one’s life." 
	"It is both exciting and frightening. " 
	"But friendship is a sacred bond between people." 
	"It is a special kind of love between us. "  
	"Many of you must have tried searching for a friend " 
	"but never found the right one.")

tokenized = sent_tokenize(txt) 
for i in tokenized: 
 
	wordsList = nltk.word_tokenize(i) 

 
	wordsList = [w for w in wordsList if not w in stop_words] 


	tagged = nltk.pos_tag(wordsList) 

	print(tagged) 


# In[ ]:




