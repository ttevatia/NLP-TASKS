#!/usr/bin/env python
# coding: utf-8

# # CLASS TASK (13-DECEMBER-2019) 

# ### TAVISH TEVATIA 17BCE2029

# In[1]:


import nltk


# In[6]:


nltk.download()


# ## BROWN CORPUS
# 

# In[3]:


from nltk.corpus import brown


# In[5]:


brown.categories()


# In[7]:


brown.words(categories='adventure')


# In[9]:


print(len(brown.words(categories='adventure')))


# ## INAUGURAL CORPUS

# In[10]:


from nltk.corpus import inaugural


# In[11]:


inaugural.fileids()


# ### LINCOLN 

# In[15]:


inaugural.words(fileids='1861-Lincoln.txt')


# In[16]:


inaugural.words(fileids='1861-Lincoln.txt')[:5]


# ### OBAMA 

# In[18]:


inaugural.words(fileids='2009-Obama.txt')


# In[21]:


inaugural.words(fileids='2009-Obama.txt')[:20]


# ## BOOK CORPUS

# In[3]:


from nltk.book import *


# ### MOBY DICK 1851

# In[4]:


f=FreqDist(text1)


# #### 20 MOST CPOMMON WORDS

# In[5]:


f.most_common(20)


# In[ ]:




