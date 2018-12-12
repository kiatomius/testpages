---
title: Summary
nav_include: 5
---

## 1. Results

### 1.1 User Model

| Model               | Train score | Test score #1 | Test score #3 |
|---------------------|-------------|---------------|---------------|
| Decision Tree       | 98.9%       | 69.9%         | 77.7%         |
| Bagging             | 100%        | 84.8%         | 54.4%         |
| Boosting            | 100%        | 88.0%         | 52.9%         |
| Random Forest       | 100%        | 84.1%         | 77.5%         |
| Logistic Regression | 97.6%       | 69.0%         | 51.7%         |
| KNN                 | 97.6%       | 64.4%         | 67.2%         |

### 1.2 User model with sentiment analysis

| Model               | Train score | Test score #1 | Test score #3 |
|---------------------|-------------|---------------|---------------|
| Decision Tree       | XXX%        | NA            | 77.7%         |
| Bagging             | XXX%        | NA            | 54.4%         |
| Boosting            | XXX%        | NA            | 52.9%         |
| Random Forest       | XXX%        | NA            | 77.5%         |
| Logistic Regression | XXX%        | NA            | 51.7%         |
| KNN                 | XXX%        | NA            | 67.2%         |

## Conclusion & Future Work 
A major challenge of predicting whether a user is bot or not is that there is no means to verify the ground truth. In our project we overcame this hurdle by taking the truth in the cresci paper (refernce in the literature review) as the true values. A couple of suggestions floated around in our discussions with the other teams and TF on ways to determine the ground truth: 
i)	We could take random samples of twitter data and manually test it. We argued against it as we were unsure of our competency in distinguishing between the two. This is further supported by the data which notes that the accuracy of human annotators ranged from 69% to 82%. 
ii)	We could use BotOrNot? which generates a score between 0 to 5, indicating the probability of whether a specific user account is automated twitter bot or not. Other than the fact that BotOrNot? is extremely time-taking, we also struggle with accuracy issues. The paper lists the accuracy of the software between 73% to 91%. 
