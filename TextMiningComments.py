#!/usr/bin/env python
# coding: utf-8

# #                                          TEXT MINING EXERCISE

# #### Import require libraries

# In[ ]:


# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import nltk
from scipy.stats.stats import pearsonr
import os

import codecs
import re
import copy
import collections
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
#from __future__ import division
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


#define working directory
os.chdir('/Applications/Documents/Application Documents/University of Phoenix/')


# #### Data Extraction : 
# Pulled required datasets into dataframes

# In[129]:


#import dataset
facebookmixer = pd.read_csv('facebook_Mixer_data.csv')
CalendarDimension = pd.read_csv('CalendarDimension.csv')
CalendarDimension_Prep= pd.read_csv('CalendarDimension_Prep.csv')
Calendar2018=pd.read_csv('CalendarDimension_2018.csv')
Comments= pd.read_csv('Commentsdata.csv')
FactsData=pd.read_csv('FactsData.csv')


# In[130]:


#Comments
Comments.info()


# In[131]:


#Facebookmixer
facebookmixer.info()


# #### Merge Students dataset with Comments data. So we can use it later for Text mining. 

# In[132]:


# Join FB And comments Data
TextMining = facebookmixer.join(Comments.set_index('Student ID'), on='Student ID')
TextMining.fillna(0, inplace=True)
TextMining.info()


# In[133]:


# Download stopwords

nltk.download('stopwords')


# In[134]:


nltk.download('all')


# #### Data Processing : Check for English stopwords

# In[135]:


from nltk.corpus import stopwords


# In[136]:


with codecs.open("Comments.csv", "r", encoding="utf-8") as f:
    Comments = f.read()


# In[137]:


esw = stopwords.words('english')
esw.append("would")


# In[138]:


word_pattern = re.compile("^\w+$")


# #### Create a token Counter Function

# In[139]:



def get_text_counter(text):
    tokens = WordPunctTokenizer().tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x:x.lower(),tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    return collections.Counter(tokens), len(tokens)


# #### Create a function to calculate the absolute frequency and relative frequency of the most common words.

# In[140]:



def make_df(counter, size):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq / size
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq, rel_freq]).T,index=index, columns=["Absolute frequency", "Relative frequency"])
    df.index.name = "Most common words"
    return df
  


# In[143]:


# Save the 10 most common words of Comments to CSV
je_counter, je_size=get_text_counter(Comments)
make_df(je_counter.most_common(10), je_size)


# #### Compare top with Merged table

# In[99]:


import numexpr
TextMining[TextMining.Comments.str.contains('time', case=True).fillna(False)]


# In[67]:


TextMining[TextMining.Comments.str.contains('work', case=True).fillna(False)]


# #### Conclusion : Eventhough top 10 words have a neutral sentiment, In the context of business problem,We notice a negative sentiment
