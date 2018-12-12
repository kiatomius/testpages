---
title: Models
notebook: Main_Model.ipynb
nav_include: 3
---

## 1. Decision Tree

### 1.1 Designing the model
A simple decision tree classifier model with maximum depth = 3 is devised as the starting point.

```python
decision_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
decision_model.fit(X_train, y_train)

y_pred_train_dec = decision_model.predict(X_train)

y_pred_test_1_dec = decision_model.predict(X_test_1)
y_pred_test_3_dec = decision_model.predict(X_test_3)

train_score_dec = accuracy_score(y_train, y_pred_train_dec) * 100

test_score_1_dec = accuracy_score(y_test_1, y_pred_test_1_dec) * 100
test_score_3_dec = accuracy_score(y_test_3, y_pred_test_3_dec) * 100
```
### 1.2 Results
    accuracy score of the training set is 98.9%
    accuracy score of the test set with social spambot #1 is 69.87891019172552%
    accuracy score of the test set with social spambot #3 is 77.69396551724138%
    
### 1.3 Exploring different depths

![png](Main_Model_files/Main_Model_22_1.png)

Depth exploration confirms that depth=3 achieves the highest test set accuracy score for both test-sets, however, for test_set_1, increasing the depth would result in higher accuracy score, with convergence at around 90%. This on the other hand results in lower score for test_set_3.

## 2. Bagging Classifier

### 2.1 Designing the model
As an extension of Decision Tree Model, we devise a bagging classifer by creating overfit model at depth=100, and the number of estimators to 100.

```python
overfit_depth = 100
N = 100

bagging_model = BaggingClassifier(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=overfit_depth), 
                                  n_estimators = N, bootstrap = True, oob_score = True)
bagging_model.fit(X_train, y_train)

y_pred_train_bag = bagging_model.predict(X_train)
y_pred_test_1_bag = bagging_model.predict(X_test_1)
y_pred_test_3_bag = bagging_model.predict(X_test_3)
train_score_bag = accuracy_score(y_train, y_pred_train_bag) * 100
test_score_1_bag = accuracy_score(y_test_1, y_pred_test_1_bag) * 100
test_score_3_bag = accuracy_score(y_test_3, y_pred_test_3_bag) * 100
```
### 2.2 Results
    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #1 is 84.81331987891019%
    accuracy score of the test set with social spambot #3 is 54.418103448275865%
    
Bagging had significantly increased the testing score for social spambot #1, however it performed poorly for the testing set #3. This could be the result of fast diminishing accuracy score of overfitting that we saw in a simple decision tree model.

## 3 Boosting

### 3.1 Designing the model
We now try to amend the overfitting problem via boosting model.

```python
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),  
                               n_estimators=800, learning_rate=0.05)

ada_model.fit(X_train, y_train)

y_pred_train_ada = ada_model.predict(X_train)
y_pred_test_1_ada = ada_model.predict(X_test_1)
y_pred_test_3_ada = ada_model.predict(X_test_3)

train_score_ada = accuracy_score(y_train, y_pred_train_ada) * 100
test_score_1_ada = accuracy_score(y_test_1, y_pred_test_1_ada) * 100
test_score_3_ada = accuracy_score(y_test_3, y_pred_test_3_ada) * 100

train_staged_score_ada = list(ada_model.staged_score(X_train, y_train))
test_staged_score_1_ada = list(ada_model.staged_score(X_test_1, y_test_1))
test_staged_score_3_ada = list(ada_model.staged_score(X_test_3, y_test_3))
```
### 3.2 Results
    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #1 is 88.04238143289606%
    accuracy score of the test set with social spambot #3 is 52.90948275862068%

Further improvement on test set with socail spambot #1 was observed but not improvement on social spambot #3 was seen.

### 3.3 Exploring the number of iteration

![png](Main_Model_files/Main_Model_25_1.png)

Above exploration of iteration shows that low accuracy could have stemmed from having too many iterations, possibly resulting in overfitting of the data. WIth appropriate number of iterations, the model has potential to increase test set #3 accuracy up to +70% level while maintaining the accuracy score for test set #1 at above 80% level.

## 4 Random Forest Model

### 4.1 Designing the model
Now we explore Random Forest model, seeking to overcome the previous overfitting problems, especially for test set #3.

```python
overfit_depth = 100
N = 100

rf_model = RandomForestClassifier(n_estimators = N, criterion='gini', 
                                  max_features='auto', max_depth = overfit_depth, bootstrap=True,
                                 oob_score=True)

rf_model.fit(X_train, y_train)
y_pred_train = rf_model.predict(X_train)

y_pred_test_1 = rf_model.predict(X_test_1)
y_pred_test_3 = rf_model.predict(X_test_3)

train_score = accuracy_score(y_train, y_pred_train) * 100

test_score_1 = accuracy_score(y_test_1, y_pred_test_1) * 100
test_score_3 = accuracy_score(y_test_3, y_pred_test_3) * 100

oobs_score = rf_model.oob_score_
```
### 4.2 Results
    accuracy score of the training set is 100.0%
    accuracy score of the test set with social spambot #1 is 84.05650857719476%
    accuracy score of the test set with social spambot #3 is 77.47844827586206%
    
Significant improvement in the accuracy score for both test sets were seen, exceeding our initial target levels, and also achieving the higher target levels achieved by Botometer.


```python
pd.Series(rf_model.feature_importances_,index=list(X_train)).sort_values().plot(kind="barh")
```
![png](Main_Model_files/Main_Model_30_1.png)

## 5. Multinominal Regression

### 5.1 Designing the model
In addition to the decision tree based classification models, we will also run traditional multinominal logistic regression models to test their classification performance.

```python
log_model = LogisticRegressionCV(fit_intercept=True, cv=5, multi_class="ovr", penalty='l2', max_iter=10000)
log_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_log = log_model.predict(X_train)

y_pred_test_1_log = log_model.predict(X_test_1)
y_pred_test_3_log = log_model.predict(X_test_3)

train_score_log = accuracy_score(y_train, y_pred_train_log) * 100

test_score_1_log = accuracy_score(y_test_1, y_pred_test_1_log) * 100
test_score_3_log = accuracy_score(y_test_3, y_pred_test_3_log) * 100
```
### 5.2 Results
    accuracy score of the training set is 97.55%
    accuracy score of the test set with social spambot #1 is 69.02119071644803%
    accuracy score of the test set with social spambot #3 is 51.724137931034484%
    
The performance of multinominal regression models fell short of decision tree based classificaiton models.

## 6 K-Nearest Neighbour Model

### 6.1 Designing the model
We will finally run K-Nearest neighbour classification model to check its performance with a range of k-values.
We first normalize the training and testing data to allow this modelling.

```python
def normalize (df):
    con_var = ['followers_count', 'listed_count', 'friends_count', 'favourites_count', 'statuses_count']

    for var in con_var:
        x = df[var]
        x = (x - x.mean())/x.std()
        df[var] = x
    
    return df

X_train_norm = normalize(X_train)
X_test_1_norm = normalize(X_test_1)
X_test_3_norm = normalize(X_test_3)
```

### 6.2 Exploring different depths
```python
kvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50]
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
![png](Main_Model_files/Main_Model_33_1.png)

From above analysis, we determine that k=3 produces optimal trainig score.

### 6.3 Results

```python
knn_model = KNeighborsClassifier(n_neighbors=10,weights = 'uniform')
knn_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_knn = knn_model.predict(X_train)

y_pred_test_1_knn = knn_model.predict(X_test_1)
y_pred_test_3_knn = knn_model.predict(X_test_3)

train_score_knn = accuracy_score(y_train, y_pred_train_knn) * 100

test_score_1_knn = accuracy_score(y_test_1, y_pred_test_1_knn) * 100
test_score_3_knn = accuracy_score(y_test_3, y_pred_test_3_knn) * 100
```
    accuracy score of the training set is 97.55%
    accuracy score of the test set with social spambot #1 is 64.42986881937436%
    accuracy score of the test set with social spambot #3 is 67.24137931034483%

We learnt that K-Nearest neighbour model performs reasonably well for both test sets, however also fall short of some of the superior performing models such as Random Forest.    


## 7. Summary

### 7.1 Comparison of model performance

| Model               | Train score | Test score #1 | Test score #3 |
|---------------------|-------------|---------------|---------------|
| Decision Tree       | 98.9%       | 69.9%         | 77.7%         |
| Bagging             | 100%        | 84.8%         | 54.4%         |
| Boosting            | 100%        | 88.0%         | 52.9%         |
| Random Forest       | 100%        | 84.1%         | 77.5%         |
| Logistic Regression | 97.6%       | 69.0%         | 51.7%         |
| KNN                 | 97.6%       | 64.4%         | 67.2%         |

* As the above chart shows, Random Forest model had the highest accuracy score for both testing sets, indicating the high predictive power as well as the robustness of its performance across different types of automates bots. 
* By separating the results between two test scores, Boosting performed best for Test Score #1, while Decision Tree with depth=3 performed best for Test score #3. While the margins of their respective performance over Random Forest are narrow, this indicates that it could be optimal to choose different models to predict certain types of automated bots.
* We have managed to achieve our initial target of beating the testing scores for Yang. et al for both testing sets. We have also beat BotOrNot? for Testing Set #1, but fell short of their performance for Testing Set #3, indicating further scope for improvement.

### 7.2 Next Steps
* We have thus far relied on _user data_ to detect automated twitter bots, basing our approach on tools proposed by various researchers. There are, however, another large field of twitter bot detection research, that are based on the analysis of tweets (instead of users).
* We believe that this field constitutes an important part in our research, and sought to further improve our model performance, with results presented in the following chapters. 