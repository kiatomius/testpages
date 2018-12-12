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

Depth exploration confirms that depth=3 achieves the highest test set accuracy score for both test-sets, however, for test_set_1, increasing the depth would result in higher accuracy score, with convergence at around 90%. This however results in lower score for test_set_3 to low 50s%.

## 2. Bagging Classifier

### 2.1 Designing the model
As an extension of Decision Tree, we devise a bagging classifer by creating overfit depth at 100, and also increasing the number of estimators to 100.

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
    
Bagging had significantly increased the testing score for social spambot #1, however it performed poorly for the testing set - social spambot #3. This could be the result of fast diminishing accuracy score of overfitting that we saw in a simple decision tree model.

## 3 Boosting

### 3.1 Designing the model
We now try to amend the abovementioned overfitting problem bia boosting model.

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

Above exploration of iteration shows that this could have stemmed from having too many number of iterations, possibly resulting in overfitting of the data. WIth appropriate number of iterations, the model has potential to increase test set #3 accuracy up to +70% level while maintaining the accuracy score for test set #1 at above 80% level.

## 4 Random Forest Model

### 4.1 Designing the model
Now we explore Random Forest model, seeking to overcome the previous overfitting problems, especially for test set 3.

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





    <matplotlib.axes._subplots.AxesSubplot at 0x195ea451e48>




![png](Main_Model_files/Main_Model_30_1.png)




```python
# Multinominal Logistic Regression

log_model = LogisticRegressionCV(fit_intercept=True, cv=5, multi_class="ovr", penalty='l2', max_iter=10000)
log_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_log = log_model.predict(X_train)

y_pred_test_1_log = log_model.predict(X_test_1)
y_pred_test_3_log = log_model.predict(X_test_3)

train_score_log = accuracy_score(y_train, y_pred_train_log) * 100

test_score_1_log = accuracy_score(y_test_1, y_pred_test_1_log) * 100
test_score_3_log = accuracy_score(y_test_3, y_pred_test_3_log) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #1 is {}%'.format(test_score_1_log))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_log))
```


    accuracy score of the training set is 97.55%
    accuracy score of the test set with social spambot #1 is 69.02119071644803%
    accuracy score of the test set with social spambot #3 is 51.724137931034484%
    



```python
# K Nearest Neighbours

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




```python
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




![png](Main_Model_files/Main_Model_33_1.png)




```python
knn_model = KNeighborsClassifier(n_neighbors=10,weights = 'uniform')
knn_model.fit(X_train, y_train.values.reshape(-1))

y_pred_train_knn = knn_model.predict(X_train)

y_pred_test_1_knn = knn_model.predict(X_test_1)
y_pred_test_3_knn = knn_model.predict(X_test_3)

train_score_knn = accuracy_score(y_train, y_pred_train_knn) * 100

test_score_1_knn = accuracy_score(y_test_1, y_pred_test_1_knn) * 100
test_score_3_knn = accuracy_score(y_test_3, y_pred_test_3_knn) * 100

print('accuracy score of the training set is {}%'.format(train_score_log))
print('accuracy score of the test set with social spambot #1 is {}%'.format(test_score_1_knn))
print('accuracy score of the test set with social spambot #3 is {}%'.format(test_score_3_knn))
```


    accuracy score of the training set is 97.55%
    accuracy score of the test set with social spambot #1 is 63.97578203834511%
    accuracy score of the test set with social spambot #3 is 68.53448275862068%
    
