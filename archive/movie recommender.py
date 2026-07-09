#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import ast 

import warnings
warnings.filterwarnings('ignore')

counter = 0 


# In[2]:


# pip install nltk


# In[3]:


print("numpy",np.__version__)
print("pandas",pd.__version__)


# In[4]:


movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')


# In[5]:


print(movies.columns)
print(credits.columns)


# In[6]:


movies.head()


# In[7]:


credits.head()


# In[8]:


mdata = movies.merge(credits,on='title')


# In[9]:


mdata.head()


# In[10]:


movies = mdata[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l   


# In[17]:


movies['genres'] = movies['genres'].apply(convert)


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[19]:


def convert3(obj):
    l=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
          l.append(i['name'])
          counter+=1
        else:
            break
    return l


# In[20]:


movies['cast'] = movies['cast'].apply(convert3)


# In[21]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
             l.append(i['name'])
             break
    return l



# In[22]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[23]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())                    


# In[24]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x ])


# In[25]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x ])


# In[26]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x ])


# In[27]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x ])


# In[28]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[29]:


movies.head()


# In[30]:


new_df = movies[['movie_id','title','tags']]


# In[31]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[32]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[33]:


new_df.head()


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[35]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[36]:


vectors


# In[37]:


# get_ipython().system('pip install nltk')


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[40]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return  " ".join(y)



# In[41]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity = cosine_similarity(vectors)


# In[45]:


similarity


# In[46]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[47]:


recommend('Avatar')


# In[48]:


recommend('Batman Begins')


# In[49]:


recommend('Superman')


# In[50]:


recommend('Harry Potter and the Goblet of Fire')


# In[51]:


recommend('Harry Brown')


# In[52]:


recommend('Tangled')


# In[53]:


recommend('John Carter')


# In[54]:


recommend('The Avengers')


# In[55]:


recommend('Harry Potter and the Goblet of Fire')


# In[56]:


import pickle


# In[57]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[58]:


new_df


# In[59]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[60]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[61]:


#import gzip
#import pickle

#with open('similarity.pkl', 'rb') as f_in:
  #  with gzip.open('similarity.pkl.gz', 'wb') as f_out:
   #     f_out.writelines(f_in)


# In[ ]:





# In[62]:


#import pickle
#mport bz2

# Save the pickle file with bz2 compression
#with bz2.open('your_file.pkl.bz2', 'wb') as f:
    ##our_data = pickle.load(f)


# In[63]:


import pickle
import gzip

# Replace 'your_file.pkl' with your actual pickle file name
with open('similarity.pkl', 'rb') as f_in:
    with gzip.open('similarity.pkl.gz', 'wb') as f_out:
        f_out.writelines(f_in)

