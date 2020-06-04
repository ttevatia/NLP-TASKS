#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import inaugural


# In[3]:


inaugural.fileids()


# In[4]:


inaugural.words(fileids='1933-Roosevelt.txt')


# # WEBTEXT CORPUS

# In[5]:


from nltk.corpus import webtext


# In[6]:


webtext.fileids()


# In[7]:


webtext.words(fileids='pirates.txt')[:10]


# In[16]:


k= webtext.fileids()


# In[46]:


for i in range (len(k)):
    print(k[i]+" "+str(webtext.words(fileids=k[i])[:20]))


# # AMERICAN NATIONAL CORPUS

# In[20]:


import nltk


# In[21]:


from nltk.book import *


# In[22]:


f = open('tweets1.txt','r')


# In[23]:


text=f.read()


# In[24]:


text1=text.split()


# In[25]:


text2=nltk.Text(text1)


# In[28]:


text2.concordance("good")


# # DIRECT EXTRATCION FORM URL

# In[30]:


from urllib import request
url="https://www.gutenberg.org/files/2554/2554-0.txt"


# In[31]:


response = request.urlopen(url)


# In[32]:


raw = response.read().decode('utf8')


# In[33]:


type(raw)


# In[34]:


len(raw)


# In[35]:


raw[:75]


# In[36]:


from nltk.tokenize import word_tokenize


# In[37]:


tokens=word_tokenize(raw)


# In[38]:


type(tokens)


# In[39]:


len(tokens)


# In[40]:


tokens[:10]


# In[41]:


txt=nltk.Text(tokens)


# In[42]:


print(txt)


# In[ ]:




