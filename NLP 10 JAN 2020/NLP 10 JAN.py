#!/usr/bin/env python
# coding: utf-8

# # I. ACCESSING LEXICAL RESOURCE

# ### 1. STOPWORDS

# In[1]:


from nltk.corpus import stopwords


# In[2]:


stopwords.words('english')


# ## 2. CMU WORDLIST 

# In[3]:


import nltk


# In[4]:


entries = nltk.corpus.cmudict.entries()


# In[5]:


len(entries)


# In[10]:


for entry in entries [10000:10050]:
    print(entry)


# ## 3. WORDNET

# In[11]:


from nltk.corpus import wordnet as wn


# In[13]:


wn.synsets('motorcar')


# In[15]:


wn.synset('car.n.01').lemma_names()


# # II. NLP PIPELINE

# In[16]:


import nltk


# In[17]:


texts=['''William Bradley Pitt (born December 18, 1963) is an American actor and film producer. 
He has received multiple awards, 
including two Golden Globe Awards for his acting,
and an Academy Award and a Primetime Emmy Award as producer under his production company, 
Plan B Entertainment.''']


# In[19]:


for text in texts:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        print(tagged_words)


# # III. IMPLEMENTING TOKENIZATION 

# ## TWITTER AWARE TOKENIZER

# In[1]:


from nltk.tokenize import TweetTokenizer


# In[4]:


text = 'the party was dope 1 :-) #dope'
twtk=TweetTokenizer()


# In[5]:


twtk.tokenize(text)


# # IV. FREQUENCY DISTRIBUTION

# ## BROWN CORPUS: 

# In[ ]:


from nltk.corpus import brown
news_text = brown.words()

