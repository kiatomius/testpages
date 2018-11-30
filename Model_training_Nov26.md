---
title: EDA Test
notebook: Model_training_Nov26.ipynb
nav_include: 1
---


```python
import json
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
```




```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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

from pandas.api.types import is_numeric_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

```




```python
from sklearn.utils import shuffle
```




```python
os.getcwd()
pd.set_option('display.max_columns', 100)
```




```python
df_gen_tweets = pd.read_csv('datasets_full.csv/genuine_accounts.csv/tweets.csv')
df_gen_users = pd.read_csv('datasets_full.csv/genuine_accounts.csv/users.csv')

print(len(df_gen_tweets)/len(df_gen_users))

df_gen = pd.merge(df_gen_tweets, df_gen_users, how = 'left',left_on='user_id', right_on='id')
```


    C:\Users\motoa\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning:
    
    Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
    
    

    817.3177892918826
    



```python
df_bot1_tweets = pd.read_csv('datasets_full.csv/social_spambots_1.csv/tweets.csv')
df_bot1_users = pd.read_csv('datasets_full.csv/social_spambots_1.csv/users.csv')
df_bot1_merge = pd.merge(df_bot1_tweets, df_bot1_users, how = 'left', left_on='user_id', right_on='id')

df_bot2_tweets = pd.read_csv('datasets_full.csv/social_spambots_2.csv/tweets.csv')
df_bot2_users = pd.read_csv('datasets_full.csv/social_spambots_2.csv/users.csv')
df_bot2_merge = pd.merge(df_bot2_tweets, df_bot2_users, how = 'left', left_on='user_id', right_on='id')

df_bot3_tweets = pd.read_csv('datasets_full.csv/social_spambots_3.csv/tweets.csv')
df_bot3_users = pd.read_csv('datasets_full.csv/social_spambots_3.csv/users.csv')
df_bot3_merge = pd.merge(df_bot3_tweets, df_bot3_users, how = 'left', left_on='user_id', right_on='id')
```


    C:\Users\motoa\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning:
    
    Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
    
    C:\Users\motoa\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning:
    
    Columns (7,10) have mixed types. Specify dtype option on import or set low_memory=False.
    
    



```python
df_bot4_tweets = pd.read_csv('datasets_full.csv/traditional_spambots_1.csv/tweets.csv')
df_bot4_users = pd.read_csv('datasets_full.csv/social_spambots_1.csv/users.csv')
df_bot4_merge = pd.merge(df_bot4_tweets, df_bot4_users, how = 'left', left_on='user_id', right_on='id')
```




```python
df_bot = pd.concat([df_bot2_merge, df_bot3_merge, df_bot4_merge])
```


    C:\Users\motoa\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning:
    
    Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
    
    



```python
len(df_gen), len(df_bot)
```





    (2839362, 1992193)





```python
df_gen['bot']=0
df_bot['bot']=1
```




```python
df_gen_no_dup = df_gen.drop_duplicates(subset = 'user_id', keep = 'first')
df_bot_no_dup = df_bot.drop_duplicates(subset = 'id_y', keep = 'first')

number = 1000

df_all = pd.concat([df_gen_no_dup.sample(number), df_bot_no_dup.sample(number)])
```


    C:\Users\motoa\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: FutureWarning:
    
    Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
    
    



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
    
    df = pd.DataFrame({'tweet':df['text'], 
                       'screen_name': df['screen_name'],
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
                       'bot_or_not':df['bot']
                       })
    
    return df
```




```python
df_clean = master_clean(df_all)
```




```python
'''
Converts features ‘followers_count’, ‘listed_count’, ‘friends_count’, 
‘favorites_count’, ‘statuses_count’ from string to int and 
normalizes their values. (for KNN algorithm implementation)
'''

def normalize (df):
    con_var = ['followers_count', 'listed_count', 'friends_count', 'favourites_count', 'statuses_count']

    for var in con_var:
        x = df[var]
        x = (x - x.mean())/x.std()
        df[var + str(2)] = x

```


(backup for removing outliers) 
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df


con_var = ['followers_count', 'listed_count', 'friends_count', 'favourites_count', 'statuses_count']

for var in con_var:
    df_clean[var] = remove_outlier(df_clean[[var]])

df_clean = df_clean.dropna()



```python
df_clean.head()
```





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
      <th>tweet</th>
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
      <th>bot_or_not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2414065</th>
      <td>You should read "poems" on #Wattpad #poetry ht...</td>
      <td>Gaara823</td>
      <td>Alex Estrada</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>229.0</td>
      <td>41.0</td>
      <td>36.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1709995</th>
      <td>Boracay! :) http://t.co/sxNH7TTXX9</td>
      <td>emeraldgayle</td>
      <td>Emerald Fabillar</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>179.0</td>
      <td>4.0</td>
      <td>339.0</td>
      <td>257.0</td>
      <td>2485.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2078683</th>
      <td>@fion_li @NiuB great place. Like a giant free ...</td>
      <td>koonnang</td>
      <td>Koonnang</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>123.0</td>
      <td>3.0</td>
      <td>702.0</td>
      <td>59.0</td>
      <td>666.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1545738</th>
      <td>RT @CBS6Cody: Good morning! Time to wake up an...</td>
      <td>CBS6Albany</td>
      <td>CBS 6 Albany - WRGB</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>27856.0</td>
      <td>605.0</td>
      <td>756.0</td>
      <td>291.0</td>
      <td>54344.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>729205</th>
      <td>this is exciting, ambulance just pulled up in ...</td>
      <td>andyinsdca</td>
      <td>andyinsdca</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>534.0</td>
      <td>22.0</td>
      <td>860.0</td>
      <td>41.0</td>
      <td>16942.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
fig, ax = plt.subplots(1,2, figsize = (15, 5))

ax[0].plot(df_clean[df_clean['bot_or_not']==1]['followers_count'],
         df_clean[df_clean['bot_or_not']==1]['friends_count'],
         'o', color = 'red', markersize = 2, alpha = 0.5, label = 'bot')

ax[0].plot(df_clean[df_clean['bot_or_not']==0]['followers_count'],
         df_clean[df_clean['bot_or_not']==0]['friends_count'],
         'o', color = 'blue',markersize = 2, alpha = 0.5, label = 'human')
ax[0].set_xlabel('followers_count')
ax[0].set_ylabel('friends_count')
ax[0].legend()

ax[1].plot(df_clean[df_clean['bot_or_not']==1]['listed_count'],
         df_clean[df_clean['bot_or_not']==1]['friends_count'],
         'o', color = 'red', markersize = 2, alpha = 0.5, label = 'bot')

ax[1].plot(df_clean[df_clean['bot_or_not']==0]['listed_count'],
         df_clean[df_clean['bot_or_not']==0]['friends_count'],
         'o', color = 'blue',markersize = 2, alpha = 0.5, label = 'human')
ax[1].set_xlabel('listed_count')
ax[1].set_ylabel('friends_count')
ax[1].legend()
```





    <matplotlib.legend.Legend at 0x252d45657f0>




![png](Model_training_Nov26_files/Model_training_Nov26_17_1.png)




```python
fig, ax = plt.subplots (2,3, figsize = (10,3))
fig.subplots_adjust(bottom = -0.8, top = 1)

size = 5

ax[0,0].plot(df_clean['friends_count'], df_clean['bot_or_not'], 'o', markersize = size, alpha =0.3)
ax[0,0].set_xlabel('friends_count')

ax[0,1].plot(df_clean['followers_count'], df_clean['bot_or_not'], 'o', markersize = size, alpha =0.3)
ax[0,1].set_xlabel('followers_count')

ax[0,2].plot(df_clean['listed_count'], df_clean['bot_or_not'], 'o', markersize = size, alpha =0.3)
ax[0,2].set_xlabel('listed_count')

ax[1,0].plot(df_clean['favourites_count'], df_clean['bot_or_not'], 'o', markersize = size, alpha =0.3)
ax[1,0].set_xlabel('favourites_count')

ax[1,1].plot(df_clean['statuses_count'], df_clean['bot_or_not'], 'o', markersize = size, alpha =0.3)
ax[1,1].set_xlabel('statuses_count')
```





    Text(0.5,0,'statuses_count')




![png](Model_training_Nov26_files/Model_training_Nov26_18_1.png)




```python
fig, ax = plt.subplots(1,2, figsize = (10, 3))
size = 0.5
alpha = 0.5

var1 = 'favourites_count'

ax[0].hist(df_clean[df_clean['bot_or_not']==1][var1],color = 'red',  alpha = alpha, label = 'bot',
           bins =np.arange(0, 100, 10))

ax[0].legend()

ax[1].hist(df_clean[df_clean['bot_or_not']==0][var1],color = 'blue', alpha = alpha, label = 'human',
          bins = np.arange(0, 10000, 100))
ax[1].legend()

ax[0].set_title(var1)
ax[1].set_title(var1)
```





    Text(0.5,1,'favourites_count')




![png](Model_training_Nov26_files/Model_training_Nov26_19_1.png)




```python
fig, ax = plt.subplots(1,2, figsize = (10, 3))
size = 0.5
alpha = 0.5

var1 = 'friends_count'

ax[0].hist(df_clean[df_clean['bot_or_not']==1][var1],color = 'red',  alpha = alpha, label = 'bot',
           bins =np.arange(0, 4000, 100))

ax[0].legend()
ax[1].hist(df_clean[df_clean['bot_or_not']==0][var1],color = 'blue', alpha = alpha, label = 'human',
          bins = np.arange(0, 4000, 100))
ax[1].legend()

ax[0].set_title(var1)
ax[1].set_title(var1)
```





    Text(0.5,1,'friends_count')




![png](Model_training_Nov26_files/Model_training_Nov26_20_1.png)




```python
fig, ax = plt.subplots(1,2, figsize = (10, 3))
size = 0.5
alpha = 0.5

var1 = 'followers_count'

ax[0].hist(df_clean[df_clean['bot_or_not']==1][var1],color = 'red',  alpha = alpha, label = 'bot',
           bins =np.arange(0, 4000, 100))

ax[0].legend()
ax[1].hist(df_clean[df_clean['bot_or_not']==0][var1],color = 'blue', alpha = alpha, label = 'human',
          bins = np.arange(0, 4000, 100))
ax[1].legend()

ax[0].set_title(var1)
ax[1].set_title(var1)
```





    Text(0.5,1,'followers_count')




![png](Model_training_Nov26_files/Model_training_Nov26_21_1.png)




```python
fig, ax = plt.subplots(1,2, figsize = (10, 3))
size = 0.5
alpha = 0.5

var1 = 'listed_count'

ax[0].hist(df_clean[df_clean['bot_or_not']==1][var1],color = 'red',  alpha = alpha, label = 'bot',
           bins =np.arange(0, 100, 1))

ax[0].legend()
ax[1].hist(df_clean[df_clean['bot_or_not']==0][var1],color = 'blue', alpha = alpha, label = 'human',
          bins = np.arange(0, 100, 1))
ax[1].legend()

ax[0].set_title(var1)
ax[1].set_title(var1)
```





    Text(0.5,1,'listed_count')




![png](Model_training_Nov26_files/Model_training_Nov26_22_1.png)




```python
fig, ax = plt.subplots(1,2, figsize = (10, 3))
size = 0.5
alpha = 0.5

var1 = 'statuses_count'

ax[0].hist(df_clean[df_clean['bot_or_not']==1][var1],color = 'red',  alpha = alpha, label = 'bot',
           bins =np.arange(0, 10000, 100))

ax[0].legend()
ax[1].hist(df_clean[df_clean['bot_or_not']==0][var1],color = 'blue', alpha = alpha, label = 'human',
          bins = np.arange(0, 10000, 100))
ax[1].legend()

ax[0].set_title(var1)
ax[1].set_title(var1)
```





    Text(0.5,1,'statuses_count')




![png](Model_training_Nov26_files/Model_training_Nov26_23_1.png)




```python
df_clean_2 = df_clean.drop(columns=['tweet', 'screen_name','name']).dropna()
df_clean_3 = shuffle(df_clean_2)
```




```python
df_X = df_clean_3.drop(columns='bot_or_not')
df_y = df_clean_3[['bot_or_not']]
```




```python
display(df_X.head())
display(df_y.head())
```



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>701774</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4520.0</td>
      <td>0.0</td>
      <td>230.0</td>
      <td>89.0</td>
      <td>416.0</td>
    </tr>
    <tr>
      <th>2104380</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>524.0</td>
      <td>1.0</td>
      <td>510.0</td>
      <td>4532.0</td>
      <td>9352.0</td>
    </tr>
    <tr>
      <th>219222</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>1085798</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>283.0</td>
      <td>1.0</td>
      <td>225.0</td>
      <td>956.0</td>
      <td>2071.0</td>
    </tr>
    <tr>
      <th>1072834</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>652.0</td>
      <td>2.0</td>
      <td>1198.0</td>
      <td>0.0</td>
      <td>9904.0</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>bot_or_not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>701774</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2104380</th>
      <td>0</td>
    </tr>
    <tr>
      <th>219222</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1085798</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1072834</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.2, shuffle = True)
```




```python
depth = 100
N = 100

rf_model = RandomForestClassifier(n_estimators = N, criterion='gini', 
                                  max_features='auto', max_depth = depth, bootstrap=True,
                                 oob_score=True)

rf_model.fit(X_train, y_train)
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_score = accuracy_score(y_train, y_pred_train) * 100
test_score = accuracy_score(y_test, y_pred_test) * 100

oobs_score = rf_model.oob_score_

print('accuracy score of the training set is {}%'.format(train_score))
print('accuracy score of the test set is {}%'.format(test_score))
```


    C:\Users\motoa\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    

    accuracy score of the training set is 100.0%
    accuracy score of the test set is 98.5%
    



```python
pd.Series(rf_model.feature_importances_,index=list(X_train)).sort_values().plot(kind="barh")
```





    <matplotlib.axes._subplots.AxesSubplot at 0x252e65d4208>




![png](Model_training_Nov26_files/Model_training_Nov26_29_1.png)




```python

```

