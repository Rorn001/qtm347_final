# QTM347 Final Project
Research Question: **How does class size affect academic performance: Evidence from primary schools in Israel**

Parents and teachers generally report that they prefer smaller classes because those involved with teaching believe that smaller classes promote student learning and offer a more pleasant environment. We will use the data of the 4th and 5th grade students of more than 2000 primary school classes in Israel. We want to use machine learning models to assess the relationship between class size and academic performance in primary school and compare the results from the paper “Using Maimonides’ Rule to Estimate the Effect of Class Size on Scholastic Achievement”, which uses linear models.

Two main questions in this project are:
- Is a small or large class more beneficial for pupils’ academic performance?
- Is the relationship between class size and educational performance linear?

```
# packages
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from ISLP import load_data
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from ISLP.models import (ModelSpec as MS, summarize, poly)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (cross_validate, KFold, ShuffleSplit)
from ISLP.models import sklearn_sm
import sklearn.linear_model as skl
from ISLP.models import ModelSpec as MS
import sklearn.model_selection as skm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
```

```
# data preprocessing
# convert dta to csv
final4 = pd.io.stata.read_stata('final4.dta')
final4.to_csv('final4.dta')
final5 = pd.io.stata.read_stata('final5.dta')
final5.to_csv('final5.dta')
```

## Approaches

**1. Discussion of predictive modeling**

To predict the correlation between class size and academic performance, the most common way is to fit a regression to the data. For example, we can implement the ordinary least square to evaluate the linear relationship between these two variables. Using more flexible predictive models, such as KNN, polynomial regression, and decision trees, can give a more accurate estimate of academic performance given the class size with lower test error. However, the regression coefficients in any of these models are not indicative of the causal impact of class size on academic performance. The most obvious reason is that students are not randomly selected into different classes; if we consider the class size as our treatment, some other factors could influence students' academic performance and correlate with the class size at the same time. Therefore, the first thing before we consider any more advanced machine learning tools is to test whether this selection bias happens in our dataset. 


![classize_score_ols](https://github.com/Rorn001/qtm347_final/assets/112023862/72dd9ed5-0678-48f6-bee2-14479878e78f)

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/858576fc-8efb-42a8-adda-36ef347bb870)

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/ee055023-3ee2-488d-819c-430a3d20868a)

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/c8cb593b-b09e-4622-9921-cead84280a2d)


By OLS, we observe that:
- There is a positive correlation between class size and test score (larger class has better performance on tests)
- Larger classes tend to have less disadvantaged students, therefore correlated with higher test score
- Larger school sizes (enrollment size) tend to have larger class sizes and higher test score
- After we add controls (enrollment size and percent of disadvantaged students), the positive correlation is reduced, but still positive, which does not correspond with our intuition.

Overall, variables such as enrollment size and percentage of disadvantaged students are the covariates that are confounding factors when we evaluate the causal impact. However, OLS still fails to reflect the intuition that smaller class sizes should promote studying. There are two potential explanations:

- The percent disadvantage variable only track the number of disadvantaged students at the school level, not at the class level within schools, so PD does not explain all the nonrandom selection of students in different size of class.

- Background info: school principals may group children who are having trouble with their schoolwork into smaller classes, but since we do not have data on the number of disadvantaged students at the class level, we are unable to control this covariate.

**2. Regression Discontinuity Design (RDD)**

At the time in Israel, there was a so-called Maimonides’ Rule that determined how large the class size in primary schools should be. It was a strictly enforced public policy so that the class size could not exceed 40; as long as it was larger than 40, the school had to break it down into 2 smaller classes.

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/2318462c-5b7f-4bbe-b0d6-62b184842cfc)

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/c2653e4a-1714-46b5-87a2-bd869bc96056)

We plot how the average class size changes as the enrollment size increases. We can observe that there is a sharp drop at exactly 40 in both 4th and 5th grade and a clear upward-sloping trend before enrollment size reaches 40. This means that schools generally have only one class before the class size reaches the limit (40) of Maimonides rule. As long as one more student enrolls, the school has to break the class into two smaller, mostly likely with an equal number of students; for example, when enrollment size increases to 41, there will be one class of 20 and one of 21. As enrollment size increases to 80, most schools will split the cohort into 3 classes of 25-27 instead of 4 classes of 20. In other words, the reduction in class size due to the Maimonides rule decreases as enrollment size increases. Therefore, when the enrollment size exceeds 120 in the graph, there are no significant patterns that reflect Maimonides's rule; the class size usually stabilizes around 30-35 even though Maimonides's rule is still in effect.

The major implication of Maimonides's rule is that the classes right above and below the cutoff will have very similar features and are good counterfactuals to each other. In other words, the decision to split a 41-student class into two smaller classes of 20 is totally exogenous of other factors. Therefore, we can select classes within a certain bandwidth near the cutoff and compare the average test scores of those below and above the cutoff. This estimate could better reflect the causal impact of the class size. 

Visually, we can compare the difference between the test scores of those who are right above and below the cutoff by fitting a line to the below-cutoff population and the above-cutoff population separately. 

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/8c493b45-692e-4106-b5a6-5cf7bc59326a)
![image](https://github.com/Rorn001/qtm347_final/assets/112023862/83480541-89c2-49d1-8850-41f17c597122)

In both fourth and fifth grades, the fitted test score of the smaller classes is higher than that of the larger classes. To numerically estimate this difference, we conduct the two-stage least square (2SLS) regression analysis used in the original paper. 

![2sls_4th](https://github.com/Rorn001/qtm347_final/assets/112023862/1be6054d-2ec7-40b4-9db2-795dd63dd331)

Here, the estimated average treatment effect is about -0.5 for the fourth grade and -0.7 for the fifth grade. The negative value empirically verifies the intuition that a smaller class size is more conducive to academic performance. Nonetheless, 2SLS uses linear regression to estimate the average test score. The potential concern could be whether the linear model can provide accurate estimates of the average test score of the classes near the cutoff. Linear regression assumes linearity between two variables, so we can imagine that if the relationship between enrollment size and test score is not linear, the estimation would be way off. Then the estimated local treatment effect would also be inaccurate since we calculate it by taking the difference of those fitted test scores. 

Hence, our next step is to test the linearity with more flexible models and implement machine learning tools when estimating the average treatment effect.

## Setup

*Variables of Interest*
- Enrollment and class sizes (X)
- Avg. test scores (y)
- % disadvantaged students 

*Polynomial Regression and k-fold cross-validation*
- subset the data into two parts: classes from schools with more than 40 students and less than 40 students
- fit the polynomial regressions on both below and above the cutoff and on both 4th and 5th Grade
- CV on both to pick the optimal degree
- Bootstrap to calculate std. and compare results with 2SLS

*KNN and k-fold cross-validation*
- subset the data into two parts: classes from schools with more than 40 students and less than 40 students
- fit the KNN regressions on both below and above the cutoff and on both 4th and 5th Grade
- CV on both to pick the optimal k
- Compare bootstrapped standard error and coefficients with 2SLS

## Results from polynomial regression

**1. Linearity test**

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/cc6942c7-35a3-4c46-989e-27c8d27a89b1)

Here, the optimal degree selected by cross-validation is 6 instead of 1, which confirms that the linearity generally does not hold. 

2. ATE in fourth grade

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/89403f8d-2980-4806-9c28-69877d6d4655)

By k-fold CV…
- Below cutoff: Degree = 2
- Above cutoff: Degree = 1

Estimated ATE:
- Model Coef = -0.98, which is larger than 2SLS Coef (-0.58) 

By Bootstrap…
- Std. = 1.5247
- mean = -1.0064
- -1.0064/1.5247, not statistically significant


**3. ATE in fifth grade**

![image](https://github.com/Rorn001/qtm347_final/assets/112023862/03610950-47d2-4307-8f35-8cc2266d060b)

By k-fold CV…
- Below cutoff: Degree = 1
- Above cutoff: Degree = 6

Estimated ATE:
- Model Coef = -3.76, which is larger than 2SLS Coef (-0.74) 

By Bootstrap…
- Std. = 1.4678
- mean = -3.790
- -3.790/1.4678, significant local treatment effect

```
# example code for calculating the ATE

below=final4[(final4['c_size']>0) & (final4['c_size']<=40)]
poly4=below[['c_size','avgmath']].dropna()
X=poly4['c_size'].values.reshape(-1, 1)
Y=poly4['avgmath']
poly_features = PolynomialFeatures(degree=2).fit_transform(X)
model1=sm.OLS(Y, poly_features).fit()
target=pd.DataFrame()
target['c_size']=np.linspace(35, 40, num=6)
X=target[['c_size']].values.reshape(-1, 1)
poly_features = PolynomialFeatures(degree=2).fit_transform(X)
below_score=np.mean(model1.predict(poly_features))

above=final4[(final4['c_size']>40) & (final4['c_size']<=80)]
poly4=above[['c_size','avgmath']].dropna()
X=poly4[['c_size']].values.reshape(-1, 1)
Y=poly4['avgmath']
poly_features = PolynomialFeatures(degree=3).fit_transform(X)
model2=sm.OLS(Y, poly_features).fit()
target=pd.DataFrame()
target['c_size']=np.linspace(40, 45, num=6)
X=target[['c_size']].values.reshape(-1, 1)
poly_features = PolynomialFeatures(degree=3).fit_transform(X)
above_score=np.mean(model2.predict(poly_features))

above_score-below_score
```

```
# example code for drawing the plot with polynomial regression
plt.figure(figsize=(10, 10))
poly=final4[(final4['c_size']>0) & (final4['c_size']<=80)]
poly=poly[['c_size','avgmath']].dropna()
X=poly['c_size']
Y=poly['avgmath']
plt.scatter(X, Y, cmap='viridis', alpha=0.2)

# above
above=final4[(final4['c_size']>40) & (final4['c_size']<=80)]
poly4=above[['c_size','avgmath']].dropna()
X=poly4['c_size'].values.reshape(-1, 1)
Y=poly4['avgmath']
poly_features = PolynomialFeatures(degree=1).fit_transform(X)
model2=sm.OLS(Y, poly_features).fit()
y_pred1=model2.predict(poly_features)
b1, a1 = model2.params
X_=np.linspace(40, 80, num=41)
plt.plot(X_, b1+a1*X_, label='Above Cutoff', color='green')

# below
below=final4[(final4['c_size']>0) & (final4['c_size']<40)]
poly4=below[['c_size','avgmath']].dropna()
X=poly4['c_size'].values.reshape(-1, 1)
Y=poly4['avgmath']
poly_features = PolynomialFeatures(degree=2).fit_transform(X)
model1=sm.OLS(Y, poly_features).fit()
y_pred2=model1.predict(poly_features)
b2, a2, a3 = model1.params
X_=np.linspace(8, 40, num=41)
plt.plot(X_, b2-0.5+X_*a2+a3*X_**2, label='Below Cutoff', color='blue')


plt.axvline(40, color='red', linestyle='--', label=f'{cutoff}')
plt.title('4th grade')
plt.xlabel('enrollment size')
plt.ylabel('average math score')
plt.legend()

plt.show()
```















