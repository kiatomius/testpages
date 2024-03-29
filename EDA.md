---
title: EDA
nav_include: 2
---
## 1. Introduction and Description of Data

We obtained the original dataset used in Cresci-2017 from Bot Repository[1]. The dataset consists of a number of datafiles, consisting of user data & tweets from genuine accounts as well as a range of different bot accounts. The background behind the choice and the methodology of obtaining the data is discussed extensively in Cresci-2017, and is summarized in the following table[2]:
 
| Dataset                 | Description                                             | Accounts | Tweets  | Year |
|-------------------------|---------------------------------------------------------|----------|---------|------|
| **genuine accounts**        |  Verified accounts that are human operated              | 3474     | 8377522 | 2011 |
| **Social Spambots #1**     | Retweeters of an Italian political candidate            | 991      | 1610176 | 2012 |
| **Social Spambots #2**      | Spammers of paid apps for mobile devices                | 3457     | 428542  | 2014 |
| **Social Spambots #3**      | Spammers of products on sale at Amazon.com              | 464      | 1418626 | 2011 |
| **Traditional Spambots #1** | Training set of spammers used by Yang et al             | 1000     | 145094  | 2009 |
| **Traditional Spambots #2** | Spammers of scam URLs                                   | 100      | 74957   | 2014 |
| **Traditional Spambots #3** | Automated accounts spamming job offers                  | 433      | 5794931 | 2013 |
| **Traditional Spambots #4** | Another group of automated accounts spamming job offers | 1128     | 133311  | 2009 |

From here, we constructed the training & testing dataset in identical manner to those presented in the paper. The methodology behind the construction of the data is summarized in the below chart. In essence, it can be described as follows:
* Training dataset is created by sampling 50% data of data from Genuine accounts, and 50% from **Traditional Spambots #1**.
* Testing dataset #1 is created by taking **all** data from Social Spambots #1, and equal number of samples from genuine accounts that were not used in creating training dataset.
* Testing dataset #2 is created by taking **all** data from Social Spambots #2, and equal number of samples from genuine accounts that were not used in creating training dataset.

<p align="center">
  <img src="index_files/data_split.png" alt="data_split" width="500"/>
</p>

* The important point that requires to be highlighted here is that Cresci-2017 approached the data preparation, NOT by creating a master dataset that consists of genuine and all types of bot accounts, and splitting to training/testing dataset, but the authors used **different bot datasets** for training set and testing set. 
* We initially took the abovementioned 'conventional approach', inspired by the data-processing techniques that we learned in CS109 course, however this resulted in generating unrealistically high accuracy scores for the test sets, therefore we reverted to the methodologies applied in Cresci-2017.

## 2. Data Cleaning

The raw dataset consists of a large number of observation types from users' tweets themselves, to number of friends, number of hashtags, number of followers and so on. In seeking to extract the information that may become powerful predictors; after widely exploring the relevant literature for varaible selection, we refered to a research by Manthan Shah and Vatsal Gopani [3] for selecting relevant variable as well as cleaning them to be fed into models.The cleaning process is presented below.

### 2.1 Narrowing columns 

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
```

### 2.2 Cleaning categorical values to binary values 

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
### 2.3 Further narrowing down predictors and applying abovedefined cleaning function

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
                       'bot_or_not':df['bot']
                       })
    
    return df
```




## 3. EDA

After cleaning the data, we explored the data for several predictors to gain base understanding of the key variables, also to form some qualitative insights over differences between human and bot data.


Full list of deinitions of user object on twitter can be found in the following URL. <https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object.html>  
Some of the key defined objects are as follows:

* __location__: The user-defined location for this account’s profile. Not necessarily a location, nor machine-parseable. This field will occasionally be fuzzily interpreted by the Search service. Example: *"San Francisco, CA"*
* __description__: The user-defined UTF-8 string describing their account. Example: *"The Real Twitter API."*
* __verified__: When true, indicates that the user has a verified account. See Verified Accounts . Example: *false*
* __followers_count__: The number of followers this account currently has. Under certain conditions of duress, this field will temporarily indicate “0”. Example: *21*
* __friends_count__: The number of users this account is following (AKA their “followings”). Under certain conditions of duress, this field will temporarily indicate “0”. Example: *32*
* __listed_count__: The number of public lists that this user is a member of. Example: "listed_count": *9274*
* __favourites_count__: The number of Tweets this user has liked in the account’s lifetime. British spelling used in the field name for historical reasons. Example: *13*

### 3.1 Exploring relationships between multiple varibles:

#### 3.1.1 **followers_count vs friends_count**
* A casual separation between the clustering of human-data and bot-data can be observed, with bots tending to have lower number of followers given friends count, while humans seem to have more even relationship between friends_count and followers_count.
* It appears that bots may have tendency to focus on increasing friends_count over other metrics. This complements the theory on bots regarding bots existing primarily to tweet, re-tweet statuses which promote their agenda to their 'friends'. This behavior is different from humans as humans (in ideal world) use twitter as a medium of exchange of information and are interested in both receiving and sending tweets.

![png](EDA_Dec_2_files/EDA_Dec_2_10_1.png)

#### 3.1.2 **listed_count vs friends_count**
* A casual separation between the clustering of human-data and bot-data can also be observed, but the separation does not seem as clear since both humans and bots have relatively similar range of 'listed_count'. This chart does however illustrate that there may be some distinct pattern in which bots register friends_count, as the variable is clearly more widely distributed compared to those of human.
* One interpretation can be that, humans have more agency and need to organize their followers into lists for easier and efficient consumption of information. Bots, one can argue, have no need for any classification of  information. 

![png](EDA_Dec_2_files/EDA_Dec_12_1.png)


#### 3.1.3 **listed_count vs followers_count**
* It is also worth noting that across certain variables, such as presented hereby (listed_count vs followers_count), it is not possible to see any separation of clustering between human data and bot data.

![png](EDA_Dec_2_files/EDA_Dec_2_10_2.png)

#### 3.1.4 **status_count vs followers_count**
* Exploration with additional variables. In general, it can be noted that bot data tend to be clusterd at certain extreme values. These are good indications that selected predictors can potentially have strong predictive power to determine automated bot users.

![png](EDA_Dec_2_files/EDA_Dec_2_10_3.png)

### 3.2 Closer exploration of individual predictors:

Now we investigate the distribution of individual predictors in order to closer analyze the difference between human users and automated bots.

#### 3.2.1 **friends_count**

![png](EDA_Dec_2_files/EDA_Dec_2_13_1.png)

* It can be noted from here that distribution of human friends_count tend to spike at around a few hundred, and naturally decline thereafter, whereas for bots, distribution also seem to spike at around few hundread, but there is another spike at around 2000 which we also saw in earlier analysis. The second spike may be a result of bots' artificial designs to register as many friends as possible for various manipulation purposes, but with a threshold set at certain limit at around 2000. The second spike could be a trend in aiding us to detect the automated bots. 

#### 3.2.2 **followers_count**

![png](EDA_Dec_2_files/EDA_Dec_2_14_1.png)

* This predictor has relatively similar distribution between humans and bots, making it possibily difficult to act as a powerful predictor, at least not on its own.

#### 3.2.3 **status_count**

![png](EDA_Dec_2_files/EDA_Dec_2_16_1.png)

* There is a striking difference between the distribution of this variable between humans and bots. It is clear that humans tweet and/or retweet far more frequently than bots, with humans' values distribute all the way up to 10,000 counts whereas bots' distribution range only up to around 200 with high concentration around 0. This stark difference could provide strong predictive support. 

#### 3.2.4 **favourites_count**

![png](EDA_Dec_2_files/EDA_Dec_2_12_1.png)

* This predictor also illustrates a stark difference between the distribution of bot data and human data. Bots mark nearly zero statuses as favorites_count whereas huamns range across 0 to 10,000, similarly to above status count: one possible interpretation is that, if bots exist to increase the number of fake followers, they don’t need to mark statuses as favorite to do either. This also provides potential to be a powerful predictor.  


Having extensively explored the data, we now move onto applying various classification methods to develop predictive models.


***
_[1] Bot Repository data download page (<https://botometer.iuni.iu.edu/bot-repository/datasets.html>)_  
_[2] Cresci-2017, page 2_  
_[3] Shad and Gopani (<https://www.youtube.com/watch?v=WYCZ6ZjfAJ0>)_
















    
