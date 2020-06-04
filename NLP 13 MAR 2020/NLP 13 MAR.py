#!/usr/bin/env python
# coding: utf-8

# # TASK 8 

# # CHUNKING
# ## NAME - TAVISH TEVATIA
# ## REG NO - 17BCE2029

# In[1]:


import nltk


# In[2]:


from nltk.tokenize import RegexpTokenizer
ex1 = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
tokenizer = RegexpTokenizer('\w+')
words =tokenizer.tokenize(ex1)
# words = nltk.word_tokenize(txt)

ex = nltk.pos_tag(words)


# In[3]:


ex


# In[4]:


#grammer to identify NounPhrase
grammer = "NP:{<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammer)


# In[ ]:





# In[5]:


result  = cp.parse(ex)


# In[6]:


print(result)


# In[7]:


result.draw()


# #pos tagg

# In[7]:


from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
iob_tagged = tree2conlltags(result)
pprint(iob_tagged)


# In[9]:


ne_tree =nltk.ne_chunk(pos_tag(word_tokenize(ex1)))
print(ne_tree)


# # **Named entity Recogition - spacy**

# In[8]:


#Spacy
import spacy


# In[11]:


nlp = spacy.load("en")


# In[9]:


nlp = spacy.load("en_core_web_sm")


# In[10]:


nlp = spacy.load("en")


# In[ ]:


ex1 = '''NLP is a subfield of computer science and artificial intelligence concerned with interactions between computers and human (natural) languages. It is used to apply machine learning algorithms to text and speech.
For example, we can use NLP to create systems like speech recognition, document summarization, machine translation, spam detection, named entity recognition, question answering, autocomplete, predictive typing and so on.
Nowadays, most of us have smartphones that have speech recognition. These smartphones use NLP to understand what is said. Also, many people use laptops which operating system has a built-in speech recognition. '''


# In[ ]:


doc = nlp(ex1)


# In[ ]:


for ent in doc.ents:
  print(ent.text,ent.label_)


# In[ ]:




