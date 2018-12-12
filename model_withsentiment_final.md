---
title: Improved models
notebook: EDA_Dec_2.ipynb
nav_include: 4
---


```python
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re 
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle
```


    /anaconda3/lib/python3.6/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d




```python
os.getcwd()
pd.set_option('display.max_columns', 100)
```




```python
#download data for genuine users

df_gen_users = pd.read_csv('datasets_full.csv/genuine_accounts.csv/users.csv')

#define columns to keep for investigation

columns_to_keep = ['id', 'name', 'screen_name', 'statuses_count', 'followers_count',
       'friends_count', 'favourites_count', 'listed_count', 'url', 'lang',
       'time_zone', 'location', 'default_profile', 'default_profile_image',
       'geo_enabled', 'profile_image_url', 'profile_banner_url',
       'profile_use_background_image', 'profile_background_image_url_https',
       'profile_text_color', 'profile_image_url_https','follow_request_sent', 
       'verified','description', 'following']

# trim the bot data and add a binary column = 1 to indicate that this data is human data

df_gen_users = df_gen_users[columns_to_keep]
df_gen_users['bot'] = 0
```




```python
#features from tweet data
columns_to_keep_t = ['retweet_count','favorite_count','num_hashtags','num_urls','num_mentions']
```




```python
#combining with tweet data
df_gen_tweets = pd.read_csv('datasets_full.csv/genuine_accounts.csv/tweets.csv')
temp = df_gen_tweets.groupby("user_id").mean()
temp = temp[columns_to_keep_t]
df_gen_user_mix = pd.merge(df_gen_users, temp, how = 'left',left_on='id', right_on='user_id')
```


    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)




```python
# download data for traditional bots for training purpose
# use trad_bot_1 since it was the main focus of the 'traditional model'

df_trad_bot1_users = pd.read_csv('datasets_full.csv/traditional_spambots_1.csv/users.csv')

# trim the bot data and add a binary column = 1 to indicate that this data is bot data
df_trad_bot1_users = df_trad_bot1_users[columns_to_keep]
df_trad_bot1_users['bot'] = 1
```




```python
#combining with tweet data
df_trad_bot1_tweets = pd.read_csv('datasets_full.csv/traditional_spambots_1.csv/tweets.csv')
temp = df_trad_bot1_tweets.groupby("user_id").mean()
temp = temp[columns_to_keep_t]
df_trad_bot1_mix = pd.merge(df_trad_bot1_users, temp, how = 'left',left_on='id', right_on='user_id')
```




```python
# calling results of sentiment analysis
df_gen_sentiment = pd.read_csv('datasets_full.csv/sentiments/df_gen_tweets_sentiments.csv')
df_trad_bot1_sentiment = pd.read_csv('datasets_full.csv/sentiments/df_trad_bot1_tweets_sentiments_spare.csv')
df_test_bot3_sentiment = pd.read_csv('datasets_full.csv/sentiments/df_bot3_tweets_sentiments.csv')
df_gen_sentiment_user=df_gen_sentiment.groupby("user_id").mean()
df_trad_bot1_sentiment_user=df_trad_bot1_sentiment.groupby("user_id").mean()
df_test_bot3_sentiment_user=df_test_bot3_sentiment.groupby("user_id").mean()
```




```python
# combine sentiment data and split the genuine data into training set and test set to avoid overlap
df_gen_user_mix = pd.merge(df_gen_user_mix, df_gen_sentiment_user, how = 'left',left_on='id', right_on='user_id')
df_trad_bot1_mix = pd.merge(df_trad_bot1_mix, df_trad_bot1_sentiment_user, how = 'left',left_on='id', right_on='user_id')

df_gen_train, df_gen_test = train_test_split(df_gen_user_mix, test_size = 0.5, shuffle = True)
```




```python
#create training dataset by concatenating the training split of genuine data and trad_bot_1 data
df_train = pd.concat([df_gen_train.sample(len(df_trad_bot1_mix)), df_trad_bot1_mix])
df_train = shuffle(df_train)
```




```python
# Create the test set
# Use social spambots #3 as the separate test data set (social spambots #1 contain Italian tweets and cannot be used for sentiment analysis)

df_test_bot3_users = pd.read_csv('datasets_full.csv/social_spambots_3.csv/users.csv')
df_test_bot3_users = df_test_bot3_users[columns_to_keep]
df_test_bot3_users['bot'] = 1

#combining with tweet data
df_test_bot3_tweets = pd.read_csv('datasets_full.csv/social_spambots_3.csv/tweets.csv')
temp2 = df_test_bot3_tweets.groupby("user_id").mean()
temp2 = temp2[columns_to_keep_t]
df_test_bot3_mix = pd.merge(df_test_bot3_users, temp2, how = 'left',left_on='id', right_on='user_id')
df_test_bot3_mix = pd.merge(df_test_bot3_mix, df_test_bot3_sentiment_user, how = 'left',left_on='id', right_on='user_id')

#create test set with 50% genuine data, and 50% social spambot data
df_test_3 = pd.concat([df_gen_test.sample(len(df_test_bot3_mix)), df_test_bot3_mix])
df_test_3 = shuffle(df_test_3)
```


    /anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (7,10) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)




```python
#define data cleaning functions

#cleaning step 1: check if screen_name has a word 'bot' in it

def screen_name_check (df):
    
    word = 'bot'
    bot_name = []
    k = 0

    for i in range (len(df)):
        if pd.isnull(df.iloc[i,:]['screen_name']):
                k = 0
        else: 
            if word in df.iloc[i,:]['screen_name']:
                k = 1
            else:
                k = 0
        bot_name.append(k)
    
    return bot_name


#cleaning step 2: check if location parameter is present

def location_check(df):
    
    loc = []

    for i in range (len(df)):
        if pd.isnull(df.iloc[i,:]['location']):
            loc.append(0)
        else:
            loc.append(1)
            
    return loc

# cleaning step 3
# Set description to 1 if it contains either of these words: 
#‘bot’, ‘robot’, ‘artificial’, ‘intelligence’, ‘neural’, ‘network’, ‘automatic’ and 0 otherwise.

def description_check(df):
    keyword = ['bot', 'robot', 'artificial', 'intelligence', 'neural', 'network', 'automatic']
    bot_des = []
    k = 0

    for i in range (len(df)):
        for keyword in keyword:
            if pd.isnull(df.iloc[i,:]['description']):
                k = 0
            else:
                if df.iloc[i,:]['description'].find(keyword) == -1:
                    k = 0
                else:
                    k = 1
        bot_des.append(k)
        
    return bot_des

#cleaning step 4:
#Set verified to 1 if the sample’s verified features contents are True and 0 otherwise.

def verified_check(df):
    ver = []

    for i in range (len(df)):
        if pd.isnull(df.iloc[i,:]['verified']):
            ver.append(0)
        else:
            ver.append(1)
    return ver

#cleaning step 5:
#Check if default profile exists or not.

def default_profile_check (df):
    
    default_profile = []

    for i in range (len(df)):
        if pd.isnull(df.iloc[i,:]['default_profile']):
            default_profile.append(0)
        else:
            default_profile.append(1)
    
    return default_profile

#cleaning step 6:
#Check if default profile image is used or not.

def default_image_check (df):
    
    default_profile_image = []

    for i in range (len(df)):
        if pd.isnull(df.iloc[i,:]['default_profile_image']):
            default_profile_image.append(0)
        else:
            default_profile_image.append(1)
    
    return default_profile_image
```




```python
def master_clean (df):
    bot_name = screen_name_check (df)
    loc = location_check (df)
    bot_des = description_check (df)
    ver = verified_check (df)
    default_profile = default_profile_check (df)
    default_profile_image = default_image_check (df)
    
    df = pd.DataFrame({'screen_name': df['screen_name'],
                       'name': df['name'],
                       'bot_in_name':bot_name,
                       'bot_in_des':bot_des,
                       'location': loc,
                       'verified': ver,
                       'default_profile': default_profile,
                       'default_profile_image': default_profile_image,
                       'followers_count': df['followers_count'],
                       'listed_count': df['listed_count'],
                       'friends_count': df['friends_count'],
                       'favourites_count': df['favourites_count'],
                       'statuses_count': df['statuses_count'],
                       'retweet_count_mean':df['retweet_count'],
                       'favorite_count_mean':df['favorite_count'],
                       'num_hashtags_mean':df['num_hashtags'],
                       'num_urls_mean':df['num_urls'],
                       'num_mentions_mean':df['num_mentions'],
                       'polarity':df['polarity'],
                       'subjectivity':df['subjectivity'],
                       'bot_or_not':df['bot']
                       })
    
    return df
```




```python
#apply the cleaning function to training set and testing set
df_train = master_clean(df_train)
df_test_3 = master_clean(df_test_3)
```




```python
#add a feature of missingness
def add_missingness(df):
    inds = np.where(df.isnull())
    inds = np.array(inds).tolist()
    inds
    inds_list = []
    for i in inds[0]:
        if i not in inds_list:
            inds_list.append(i)
    df["Missingness"] = 0
    for i in inds_list:
        df.iloc[i,-1] = 1
    return df
```




```python
df_train_2=add_missingness(df_train)
df_test_3_2=add_missingness(df_test_3)
```




```python
# index of rows which contain NaN
inds = np.where(df_train_2.isna())
inds = np.array(inds).tolist()
inds_list = []
for i in inds[0]:
    if i not in inds_list:
        inds_list.append(i)
```




```python
#prepare for imputation
columns_to_keep_nan = ['retweet_count_mean', 'favorite_count_mean', 'num_hashtags_mean', 'num_urls_mean', 'num_mentions_mean', 'polarity','subjectivity']
```




```python
#linear regression imputation
from sklearn.linear_model import LinearRegression

def linear_imputation(df,columns_withna):
    df2=df.copy()
    df3=df.copy()
    df2=df2.drop(columns=["screen_name","name"])
    df_dropna=df2.dropna()
    for feature in columns_withna:
        x_set = df_dropna[df_dropna.columns.difference(columns_withna)]
        y_set = df_dropna[feature]
        
        linear_reg = LinearRegression()
        linear_reg.fit(x_set,y_set)
        ytest_hat = linear_reg.predict(df2[df2.columns.difference(columns_withna)])
        df3[feature]=ytest_hat
    df3.update(df)
    return df3
```




```python
df_train_3 = linear_imputation(df_train_2, columns_to_keep_nan)
df_test_3_3 = linear_imputation(df_test_3_2, columns_to_keep_nan)
```




```python
print(df_train_3.shape)
display(df_train_3.head())
print(df_test_3_3.shape)
display(df_test_3_3.head())
```


    (2000, 22)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>name</th>
      <th>bot_in_name</th>
      <th>bot_in_des</th>
      <th>location</th>
      <th>verified</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>followers_count</th>
      <th>listed_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>retweet_count_mean</th>
      <th>favorite_count_mean</th>
      <th>num_hashtags_mean</th>
      <th>num_urls_mean</th>
      <th>num_mentions_mean</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>bot_or_not</th>
      <th>Missingness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211</th>
      <td>astefanisilva</td>
      <td>Amanda</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2630</td>
      <td>14</td>
      <td>3855</td>
      <td>0</td>
      <td>588</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.394558</td>
      <td>0.112245</td>
      <td>0.072187</td>
      <td>0.112252</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>JuliannacaMadyi</td>
      <td>Julianna Madyson</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>393</td>
      <td>1</td>
      <td>1471</td>
      <td>0</td>
      <td>85</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.329412</td>
      <td>0.011765</td>
      <td>0.092592</td>
      <td>0.371703</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>nordestefureve</td>
      <td>Fox</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>586</td>
      <td>0</td>
      <td>1989</td>
      <td>0</td>
      <td>52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.807692</td>
      <td>0.211538</td>
      <td>0.012821</td>
      <td>0.025641</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>647</th>
      <td>hillaryapplebee</td>
      <td>Hillary Applebee</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1061</td>
      <td>4</td>
      <td>1874</td>
      <td>0</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.160000</td>
      <td>0.480000</td>
      <td>1.200000</td>
      <td>0.194369</td>
      <td>0.452571</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>871</th>
      <td>EduKalore</td>
      <td>eduardo rebellato</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>4</td>
      <td>911</td>
      <td>1</td>
      <td>54</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.203704</td>
      <td>0.425926</td>
      <td>0.185185</td>
      <td>0.012346</td>
      <td>0.024691</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    (928, 22)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>name</th>
      <th>bot_in_name</th>
      <th>bot_in_des</th>
      <th>location</th>
      <th>verified</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>followers_count</th>
      <th>listed_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>retweet_count_mean</th>
      <th>favorite_count_mean</th>
      <th>num_hashtags_mean</th>
      <th>num_urls_mean</th>
      <th>num_mentions_mean</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>bot_or_not</th>
      <th>Missingness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>Alps_Sarsis</td>
      <td>Alps Sarsis</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>87</td>
      <td>3</td>
      <td>195</td>
      <td>66</td>
      <td>1523</td>
      <td>520.595854</td>
      <td>0.252169</td>
      <td>0.347316</td>
      <td>0.191161</td>
      <td>0.746502</td>
      <td>0.093155</td>
      <td>0.286208</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>402</th>
      <td>New_EnglandNews</td>
      <td>Tom Fisher</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>856</td>
      <td>8</td>
      <td>939</td>
      <td>0</td>
      <td>24161</td>
      <td>0.008736</td>
      <td>0.010608</td>
      <td>0.000312</td>
      <td>0.074571</td>
      <td>0.070827</td>
      <td>0.177714</td>
      <td>0.377957</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>131</th>
      <td>idigjamesbond1</td>
      <td>idigjamesbond</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1018</td>
      <td>6</td>
      <td>1943</td>
      <td>1</td>
      <td>13095</td>
      <td>0.006169</td>
      <td>0.012338</td>
      <td>0.000000</td>
      <td>0.465762</td>
      <td>0.000000</td>
      <td>0.178620</td>
      <td>0.370753</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Iloveseniorfitn</td>
      <td>Love Senior Fitness</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1037</td>
      <td>10</td>
      <td>1799</td>
      <td>1</td>
      <td>12277</td>
      <td>0.004938</td>
      <td>0.012963</td>
      <td>0.000000</td>
      <td>0.457407</td>
      <td>0.000000</td>
      <td>0.184388</td>
      <td>0.377921</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2211</th>
      <td>geithsy</td>
      <td>끄 ♡</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>231</td>
      <td>1</td>
      <td>92</td>
      <td>287</td>
      <td>81721</td>
      <td>1366.627403</td>
      <td>0.148773</td>
      <td>0.391821</td>
      <td>0.366675</td>
      <td>0.898962</td>
      <td>0.076938</td>
      <td>0.332585</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#########
#Model by dropping NaN
```




```python
df_train_dn = df_train.dropna()
print(df_train_dn.shape)
display(df_train_dn.head())
df_test_3_dn = df_test_3.dropna()
print(df_test_3_dn.shape)
display(df_test_3_dn.head())
```


    (1120, 22)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>name</th>
      <th>bot_in_name</th>
      <th>bot_in_des</th>
      <th>location</th>
      <th>verified</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>followers_count</th>
      <th>listed_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>retweet_count_mean</th>
      <th>favorite_count_mean</th>
      <th>num_hashtags_mean</th>
      <th>num_urls_mean</th>
      <th>num_mentions_mean</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>bot_or_not</th>
      <th>Missingness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>211</th>
      <td>astefanisilva</td>
      <td>Amanda</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2630</td>
      <td>14</td>
      <td>3855</td>
      <td>0</td>
      <td>588</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.394558</td>
      <td>0.112245</td>
      <td>0.072187</td>
      <td>0.112252</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>JuliannacaMadyi</td>
      <td>Julianna Madyson</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>393</td>
      <td>1</td>
      <td>1471</td>
      <td>0</td>
      <td>85</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.329412</td>
      <td>0.011765</td>
      <td>0.092592</td>
      <td>0.371703</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>nordestefureve</td>
      <td>Fox</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>586</td>
      <td>0</td>
      <td>1989</td>
      <td>0</td>
      <td>52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.807692</td>
      <td>0.211538</td>
      <td>0.012821</td>
      <td>0.025641</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>647</th>
      <td>hillaryapplebee</td>
      <td>Hillary Applebee</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1061</td>
      <td>4</td>
      <td>1874</td>
      <td>0</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.160000</td>
      <td>0.480000</td>
      <td>1.200000</td>
      <td>0.194369</td>
      <td>0.452571</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>871</th>
      <td>EduKalore</td>
      <td>eduardo rebellato</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>4</td>
      <td>911</td>
      <td>1</td>
      <td>54</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.203704</td>
      <td>0.425926</td>
      <td>0.185185</td>
      <td>0.012346</td>
      <td>0.024691</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    (386, 22)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>name</th>
      <th>bot_in_name</th>
      <th>bot_in_des</th>
      <th>location</th>
      <th>verified</th>
      <th>default_profile</th>
      <th>default_profile_image</th>
      <th>followers_count</th>
      <th>listed_count</th>
      <th>friends_count</th>
      <th>favourites_count</th>
      <th>statuses_count</th>
      <th>retweet_count_mean</th>
      <th>favorite_count_mean</th>
      <th>num_hashtags_mean</th>
      <th>num_urls_mean</th>
      <th>num_mentions_mean</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>bot_or_not</th>
      <th>Missingness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>idigjamesbond1</td>
      <td>idigjamesbond</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1018</td>
      <td>6</td>
      <td>1943</td>
      <td>1</td>
      <td>13095</td>
      <td>0.006169</td>
      <td>0.012338</td>
      <td>0.0</td>
      <td>0.465762</td>
      <td>0.0</td>
      <td>0.178620</td>
      <td>0.370753</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Iloveseniorfitn</td>
      <td>Love Senior Fitness</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1037</td>
      <td>10</td>
      <td>1799</td>
      <td>1</td>
      <td>12277</td>
      <td>0.004938</td>
      <td>0.012963</td>
      <td>0.0</td>
      <td>0.457407</td>
      <td>0.0</td>
      <td>0.184388</td>
      <td>0.377921</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>JasonJacksonLeo</td>
      <td>Jason Leo</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>588</td>
      <td>4</td>
      <td>1324</td>
      <td>1</td>
      <td>11159</td>
      <td>0.008679</td>
      <td>0.020769</td>
      <td>0.0</td>
      <td>0.464042</td>
      <td>0.0</td>
      <td>0.178078</td>
      <td>0.375232</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>tommyboyceandbo</td>
      <td>Boyce &amp; Bobby</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1215</td>
      <td>7</td>
      <td>733</td>
      <td>0</td>
      <td>12915</td>
      <td>0.004360</td>
      <td>0.017440</td>
      <td>0.0</td>
      <td>0.448147</td>
      <td>0.0</td>
      <td>0.171472</td>
      <td>0.374131</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>327</th>
      <td>Whitforddig</td>
      <td>Claris Whitford</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>900</td>
      <td>1</td>
      <td>384</td>
      <td>0</td>
      <td>11078</td>
      <td>0.002795</td>
      <td>0.016460</td>
      <td>0.0</td>
      <td>0.463975</td>
      <td>0.0</td>
      <td>0.175562</td>
      <td>0.365729</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Prepare dataset to feed into the model

# X_train from df_train
X_train = df_train_3.drop(columns=['bot_or_not', 'screen_name', 'name'])
X_train_dn = df_train_dn.drop(columns=['bot_or_not', 'screen_name', 'name'])

# y_trian from df_train
y_train = df_train_3[['bot_or_not']]
y_train_dn = df_train_dn[['bot_or_not']]

# X_test_3 from df_test_3
X_test_3 = df_test_3_3.drop(columns=['bot_or_not', 'screen_name', 'name'])
X_test_dn = df_test_3_dn.drop(columns=['bot_or_not', 'screen_name', 'name'])

# y_test_3 from df_test_3
y_test_3 = df_test_3_3[['bot_or_not']]
y_test_dn = df_test_3_dn[['bot_or_not']]
```




```python
# standardize dataset

def standardize (df,df_train):
    con_var = ['followers_count', 'listed_count', 'friends_count', 'favourites_count', 'statuses_count',
               'retweet_count_mean','favorite_count_mean','num_hashtags_mean','num_urls_mean','num_mentions_mean',
              'polarity','subjectivity']

    for var in con_var:
        x = df[var]
        x = (x - x.mean())/x.std()
        df[var] = x
    
    return df

X_train_norm = standardize(X_train,X_train)
X_test_3_norm = standardize(X_test_3,X_train)
X_train_norm_dn = standardize(X_train_dn,X_train_dn)
X_test_3_norm_dn = standardize(X_test_dn,X_train_dn)
```




```python
# Simple Decision Tree with linear imputation

decision_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
decision_model.fit(X_train, y_train)

y_pred_train_dec = decision_model.predict(X_train_norm)
y_pred_test_3_dec = decision_model.predict(X_test_3_norm)

train_score_dec = accuracy_score(y_train, y_pred_train_dec) * 100
test_score_3_dec = accuracy_score(y_test_3, y_pred_test_3_dec) * 100

print('accuracy score of the training set is {}%'.format(train_score_dec))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_dec))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 49.0301724137931%




```python
# Simple Decision Tree with dropna

decision_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
decision_model.fit(X_train_dn, y_train_dn)

y_pred_train_dec = decision_model.predict(X_train_norm_dn)
y_pred_test_3_dec = decision_model.predict(X_test_3_norm_dn)

train_score_dec = accuracy_score(y_train_dn, y_pred_train_dec) * 100
test_score_3_dec = accuracy_score(y_test_dn, y_pred_test_3_dec) * 100

print('accuracy score of the training set is {}%'.format(train_score_dec))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_dec))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 95.07772020725389%




```python
depth = np.arange(2,30,1)
decision_score_mean=[]
decision_score_std=[]
test_1_score = []
test_3_score = []

for i in depth:
    decision_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=i)
    decision_model.fit(X_train_norm, y_train)
    score = cross_val_score(estimator = decision_model, X = X_train_norm, y = y_train, cv = 5)
    decision_score_mean.append(score.mean())
    decision_score_std.append(score.std())
    test_3_score.append(accuracy_score(y_test_3, decision_model.predict(X_test_3_norm)))
```




```python
fig, ax = plt.subplots(1,2, figsize = (15,5))

ax[0].plot(depth, decision_score_mean, '-*')
ax[0].fill_between(
    depth,
    np.array(decision_score_mean) - 2 * np.array(decision_score_std),
    np.array(decision_score_mean) + 2 * np.array(decision_score_std),
    alpha=.3)
ax[0].set_title('validation accuracy vs depth')
ax[0].set_xlabel('max depth')
ax[0].set_ylabel('validation accuracy +- 2 std')

ax[1].plot(depth, test_3_score)
ax[1].set_title('spambots3 test set accuracy vs depth')
ax[1].set_xlabel('max depth')
ax[1].set_ylabel('test set accuracy score')
```





    Text(0,0.5,'test set accuracy score')




![png](model_withsentiment_final_files/model_withsentiment_final_28_1.png)




```python
# bagging with linear imputation
overfit_depth = 100
N = 100

bagging_model = BaggingClassifier(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=overfit_depth), 
                                  n_estimators = N, bootstrap = True, oob_score = True)
bagging_model.fit(X_train_norm, y_train)

y_pred_train_bag = bagging_model.predict(X_train_norm)
y_pred_test_3_bag = bagging_model.predict(X_test_3_norm)

train_score_bag = accuracy_score(y_train, y_pred_train_bag) * 100
test_score_3_bag = accuracy_score(y_test_3, y_pred_test_3_bag) * 100

print('accuracy score of the training set is {}%'.format(train_score_bag))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_bag))
```


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 94.07327586206897%




```python
# bagging with dropna
overfit_depth = 100
N = 100

bagging_model = BaggingClassifier(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=overfit_depth), 
                                  n_estimators = N, bootstrap = True, oob_score = True)
bagging_model.fit(X_train_norm_dn, y_train_dn)

y_pred_train_bag = bagging_model.predict(X_train_norm_dn)
y_pred_test_3_bag = bagging_model.predict(X_test_3_norm_dn)

train_score_bag = accuracy_score(y_train_dn, y_pred_train_bag) * 100
test_score_3_bag = accuracy_score(y_test_dn, y_pred_test_3_bag) * 100

print('accuracy score of the training set is {}%'.format(train_score_bag))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_bag))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 95.07772020725389%


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)




```python
#ada boosting with linear imputation
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),  
                               n_estimators=800, learning_rate=0.05)

ada_model.fit(X_train_norm, y_train)

y_pred_train_ada = ada_model.predict(X_train_norm)
y_pred_test_3_ada = ada_model.predict(X_test_3_norm)

train_score_ada = accuracy_score(y_train, y_pred_train_ada) * 100
test_score_3_ada = accuracy_score(y_test_3, y_pred_test_3_ada) * 100

print('accuracy score of the training set is {}%'.format(train_score_ada))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_ada))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 82.4353448275862%


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)




```python
#ada boosting with dropna
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),  
                               n_estimators=800, learning_rate=0.05)

ada_model.fit(X_train_norm_dn, y_train_dn)

y_pred_train_ada = ada_model.predict(X_train_norm_dn)
y_pred_test_3_ada = ada_model.predict(X_test_3_norm_dn)

train_score_ada = accuracy_score(y_train_dn, y_pred_train_ada) * 100
test_score_3_ada = accuracy_score(y_test_dn, y_pred_test_3_ada) * 100

print('accuracy score of the training set is {}%'.format(train_score_ada))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_ada))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 95.07772020725389%


    /anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)




```python
#Random Forests with linear imputation
overfit_depth = 100
N = 100

rf_model = RandomForestClassifier(n_estimators = N, criterion='gini', 
                                  max_features='auto', max_depth = overfit_depth, bootstrap=True,
                                 oob_score=True)
rf_model.fit(X_train_norm, y_train)

y_pred_train = rf_model.predict(X_train_norm)
y_pred_test_3 = rf_model.predict(X_test_3_norm)

train_score = accuracy_score(y_train, y_pred_train) * 100
test_score_3 = accuracy_score(y_test_3, y_pred_test_3) * 100

oobs_score = rf_model.oob_score_

print('accuracy score of the training set is {}%'.format(train_score))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3))
```


    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 83.1896551724138%




```python
#showing significant features
pd.Series(rf_model.feature_importances_,index=list(X_train_norm)).sort_values().plot(kind="barh")
```





    <matplotlib.axes._subplots.AxesSubplot at 0x1a228d3080>




![png](model_withsentiment_final_files/model_withsentiment_final_34_1.png)




```python
#Random Forests with dropna
overfit_depth = 100
N = 100

rf_model = RandomForestClassifier(n_estimators = N, criterion='gini', 
                                  max_features='auto', max_depth = overfit_depth, bootstrap=True,
                                 oob_score=True)
rf_model.fit(X_train_norm_dn, y_train_dn)

y_pred_train = rf_model.predict(X_train_norm_dn)
y_pred_test_3 = rf_model.predict(X_test_3_norm_dn)

train_score = accuracy_score(y_train_dn, y_pred_train) * 100
test_score_3 = accuracy_score(y_test_dn, y_pred_test_3) * 100

oobs_score = rf_model.oob_score_

print('accuracy score of the training set is {}%'.format(train_score))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3))
```


    /anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:8: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 96.6321243523316%




```python
#showing significant features
pd.Series(rf_model.feature_importances_,index=list(X_train_norm_dn)).sort_values().plot(kind="barh")
```





    <matplotlib.axes._subplots.AxesSubplot at 0x1a21919ef0>




![png](model_withsentiment_final_files/model_withsentiment_final_36_1.png)




```python
# Multinominal Logistic Regression with linear imputation

log_model = LogisticRegressionCV(fit_intercept=True, cv=5, multi_class="ovr", penalty='l2', max_iter=10000)
log_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_log = log_model.predict(X_train_norm)
y_pred_test_3_log = log_model.predict(X_test_3_norm)

train_score_log = accuracy_score(y_train, y_pred_train_log) * 100
test_score_3_log = accuracy_score(y_test_3, y_pred_test_3_log) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_log))
```


    accuracy score of the training set is 99.95%
    accuracy score of the test set with social spambot #3 is 95.6896551724138%




```python
# Multinominal Logistic Regression with dropna

log_model = LogisticRegressionCV(fit_intercept=True, cv=5, multi_class="ovr", penalty='l2', max_iter=10000)
log_model.fit(X_train_dn, y_train_dn.values.reshape(-1))

y_pred_train_log = log_model.predict(X_train_norm_dn)
y_pred_test_3_log = log_model.predict(X_test_3_norm_dn)

train_score_log = accuracy_score(y_train_dn, y_pred_train_log) * 100
test_score_3_log = accuracy_score(y_test_dn, y_pred_test_3_log) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_log))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 95.59585492227978%




```python
#kNN  with linear imputation
kvals = [1, 2, 5, 7, 10, 15, 20, 25, 30, 50]
knn_score_train = []

for i in kvals:
    model_knn = KNeighborsClassifier(n_neighbors=i, weights = 'uniform')
    train_score = cross_val_score(model_knn, X = X_train_norm, y = y_train.values.reshape(-1), cv=5)
    knn_score_train.append(train_score.mean())

fig, ax = plt.subplots(1,1, figsize = (12,5))

ax.plot(kvals, knn_score_train)
ax.set_title("Train Set Score")
ax.set_xlabel("kvals")
ax.set_ylabel("Mean Accuracy Score")
```





    Text(0,0.5,'Mean Accuracy Score')




![png](model_withsentiment_final_files/model_withsentiment_final_39_1.png)




```python
knn_model = KNeighborsClassifier(n_neighbors=10,weights = 'uniform')
knn_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_3_knn = knn_model.predict(X_test_3)

train_score_knn = accuracy_score(y_train, y_pred_train_knn) * 100
test_score_3_knn = accuracy_score(y_test_3, y_pred_test_3_knn) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_knn))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 92.88793103448276%




```python
#kNN  with dropna
kvals = [1, 2, 5, 7, 10, 15, 20, 25, 30, 50]
knn_score_train = []

for i in kvals:
    model_knn = KNeighborsClassifier(n_neighbors=i, weights = 'uniform')
    train_score = cross_val_score(model_knn, X = X_train_norm_dn, y = y_train_dn.values.reshape(-1), cv=5)
    knn_score_train.append(train_score.mean())

fig, ax = plt.subplots(1,1, figsize = (12,5))

ax.plot(kvals, knn_score_train)
ax.set_title("Train Set Score")
ax.set_xlabel("kvals")
ax.set_ylabel("Mean Accuracy Score")
```





    Text(0,0.5,'Mean Accuracy Score')




![png](model_withsentiment_final_files/model_withsentiment_final_41_1.png)




```python
knn_model = KNeighborsClassifier(n_neighbors=10,weights = 'uniform')
knn_model.fit(X_train_dn, y_train_dn.values.reshape(-1))

y_pred_train_knn = knn_model.predict(X_train_norm_dn)
y_pred_test_3_knn = knn_model.predict(X_test_3_norm_dn)

train_score_knn = accuracy_score(y_train_dn, y_pred_train_knn) * 100
test_score_3_knn = accuracy_score(y_test_dn, y_pred_test_3_knn) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_knn))
```


    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #3 is 91.19170984455958%

