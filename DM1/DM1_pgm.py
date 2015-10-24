
# coding: utf-8

## DM1: Probabilistic Graphical Models

##### 2. Linear Classification

# In[1]:

get_ipython().magic(u'matplotlib inline')

from __future__ import division

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 13,9


###### The data is loaded and displayed

# In[2]:

classA = pd.read_csv('classificationA.train', sep='\t', header=None)
classB = pd.read_csv('classificationB.train', sep='\t', header=None)
classC = pd.read_csv('classificationC.train', sep='\t', header=None)

classA_test = pd.read_csv('classificationA.test', sep='\t', header=None)
classB_test = pd.read_csv('classificationB.test', sep='\t', header=None)
classC_test = pd.read_csv('classificationC.test', sep='\t', header=None)

def plot(df, title=None):
    ax = df.loc[df[2] == 1].plot(kind='scatter', x=0, y=1, color='blue', label='Gaussian 1')
    df.loc[df[2] == 0].plot(kind='scatter', x=0, y=1, ax=ax, color='green', label='Gaussian 2')
    if title:
        plt.title(title)
    plt.show()

plot(classA, 'classificationA train dataset')
plot(classB, 'classificationB train dataset')
plot(classC, 'classificationC train dataset')


###### 1. Generative model (LDA)

# (c) Data in R^2 along with the decision boundary line

# In[3]:

def lda(df):
    # MLE Parameters:
    pi = df[2].mean()
    mu0 = df[[0, 1]][df[2] == 0].mean()
    mu1 = df[[0, 1]][df[2] == 1].mean()
    
    Y = df[2]
    X = df[[0, 1]]
    A = np.multiply((X - mu1).T, Y).T
    B = np.multiply((X - mu0).T, (1 - Y)).T
    sigma = (np.dot(A.T, A) + np.dot(B.T, B)) / len(X-2)
    
    # Decision boudary
    sigma_inv = np.linalg.inv(sigma)
    beta = np.dot(sigma_inv, mu1 - mu0)
    gamma =  math.log(pi / (1 - pi)) - 0.5 * np.dot(np.dot((mu1 - mu0).T, sigma_inv), mu1 + mu0)
    return beta[0], beta[1], gamma


# In[4]:

def plot_lda(df, title=None):
    coefs = lda(df)
    
    # Get the separating hyperplane
    a = - coefs[0] / coefs[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (coefs[2] / coefs[1])
    ax = df.loc[df[2] == 1].plot(kind='scatter', x=0, y=1, color='blue', label='Gaussian 1')
    df.loc[df[2] == 0].plot(kind='scatter', x=0, y=1, ax=ax, color='green', label='Gaussian 2')
    
    if title:
        plt.title(title)
    plt.plot(xx,yy)
    plt.show()
    
plot_lda(classA, 'Classification on classificationA with LDA')
plot_lda(classB, 'Classification on classificationB with LDA')
plot_lda(classC, 'Classification on classificationC with LDA')


###### 2. Logistic regression

# I didn't manage to make to make the computation works, the solution explodes then crash
# for the next question I'll use the logistic regression provided in the sklearn package

# In[5]:

from sklearn.preprocessing import StandardScaler

def sig(x):
    a = 1 / (1 + np.exp(-x))
    return a
        
def logistic(df):
    Y = df[2].values
    X = df[[0,1]].values
    
    # preprocessing
    X = StandardScaler().fit_transform(X)
    
    # Add constant term
    # a = np.empty((len(X), 1)); a.fill(1)
    # X = np.append(X, a, 1)
    
    l = [np.array([0,0])]
    w = np.array([-2, -1])
    
    while np.linalg.norm(w - l[-1]) > 0.1:
        l.append(w)
        n = np.array([sig(np.dot(w, x)) * (1 - sig(np.dot(w, x))) for x in X])
        diag = np.diag(n)
        A = np.linalg.inv(np.dot(np.dot(X.T, diag), X))
        w = np.add(w, np.dot(np.dot(A, X.T), np.subtract(Y, n)))
        print w
    return w

#logistic(classA)

from sklearn.linear_model import LogisticRegression

def plot_logistic(df, title=None):# Get the separating hyperplane
    lr = LogisticRegression().fit(df[[0,1]].values, df[2].values)
    coefs = lr.coef_.tolist()[0] +  lr.intercept_.tolist()
    print coefs
    
    a = - coefs[0] / coefs[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (coefs[2] / coefs[1])
    ax = df.loc[df[2] == 1].plot(kind='scatter', x=0, y=1, color='blue', label='Gaussian 1')
    df.loc[df[2] == 0].plot(kind='scatter', x=0, y=1, ax=ax, color='green', label='Gaussian 2')
    
    if title:
        plt.title(title)
    plt.plot(xx,yy)
    plt.show()
    
plot_logistic(classA, 'Classification on classificationA with logistic regression')
plot_logistic(classB, 'Classification on classificationB with logistic regression')
plot_logistic(classC, 'Classification on classificationC with logistic regression')

[-2.1210599745149095, -1.5463123105017398, -0.251685765491529]
[-1.5035813963824607, 0.8748750392434941, 0.9920836137683351]
[-1.997651538438316, 0.5548643102693468, 0.7192013887106192]


# (a). Numerical values of the parameters learnt by logistic regression:
# 
# _ classifcationA.train: w = (-2.1210599745149095 -1.5463123105017398 -0.251685765491529)
# 
# _ classificationB.train: w = (-1.5035813963824607, 0.8748750392434941, 0.9920836137683351)
# 
# _ classificationC.train: w = (-1.997651538438316, 0.5548643102693468, 0.7192013887106192)

###### 3. Linear regression

# In[6]:

# Linear regression
def linear(df):
    Y = df[2].values
    X = df[[0,1]].values
    a = np.empty((len(X), 1)); a.fill(1)
    X = np.append(X, a, 1)
    
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    return w

print linear(classA)
print linear(classB)
print linear(classC)


# (a). Numerical values of the parameters learnt by linear regression:
# 
# _ classifcationA.train: w = (-0.2640075  -0.37259311  0.49229204)
# 
# _ classificationB.train: w = (-0.10424575  0.05179118  0.50005043)
# 
# _ classificationC.train: w = (-0.12769333 -0.01700142  0.50839982)

# (b). Data in R^2 along with the decision boundary line

# In[7]:

def plot_linear(df, title=None):
    coefs = linear(df)

    # get the separating hyperplane
    a = - coefs[0] / coefs[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (coefs[2] - 0.5) / coefs[1]
    ax = df.loc[df[2] == 1].plot(kind='scatter', x=0, y=1, color='blue', label='Gaussian 1')
    df.loc[df[2] == 0].plot(kind='scatter', x=0, y=1, ax=ax, color='green', label='Gaussian 2')
    plt.plot(xx,yy)
    if title:
        plt.title(title)
    plt.show()
    
    
plot_linear(classA, 'Classification with linear regression on classicationA')
plot_linear(classB, 'Classification with linear regression on classicationB')
plot_linear(classC, 'Classification with linear regression on classicationC')


###### 4. Benchmarking

# In[8]:

# Compute misclassification error:

def compute(df, w, mode=None):
    Y = df[2].values
    X = df[[0,1]].values
    
    # Add constant term
    a = np.empty((len(X), 1)); a.fill(1)
    X = np.append(X, a, 1)
    
    threshold = 0 if mode == 'lda' else 1/2
    pred = np.dot(X, w) > threshold
    
    return 1 - (Y == pred).mean()


# In[9]:

classes = {
    'classA': classA,
    'classB': classB,
    'classC': classC,
}

test = {
    'classA': classA_test,
    'classB': classB_test,
    'classC': classC_test,
}

classifiers = {
    'lda': lda,
    'linear': linear,
}

print 'Missclassification error is:\n'
for klass, data in classes.iteritems():
    for name, classifier in classifiers.iteritems():
        weights = classifier(data)
        error_train = compute(data, weights, mode=name)
        error_test = compute(test[klass], weights)

        print '{} on the train data and {} on the test data for {} on {}'.format(
                error_train, error_test, name, klass)
    print '\n'


# In[16]:

print 'Missclassification error is:\n'
for klass, data in classes.iteritems():
    lr = LogisticRegression().fit(data[[0,1]].values, data[2].values)
    error_train = 1 - (lr.predict(data[[0,1]])  == data[2]).mean()
    error_test =  1 - (lr.predict(test[klass][[0,1]]) == test[klass][2]).mean()

    print '{} on the train data and {} on the test data for logistic regression on {}'.format(
            error_train, error_test, klass)
    print '\n'


# (b). As we could expect after looking at the graphs, the different methods show the same result, indeed they are all trying to find the best line to seperate the data. The methods make different assumption but in the end have more or less the same results

###### 5. QDA model

# In[11]:

def qda(df):
    # MLE Parameters:
    pi = df[2].mean()
    mu0 = df[[0, 1]][df[2] == 0].mean()
    mu1 = df[[0, 1]][df[2] == 1].mean()
    
    Y = df[2]
    X0 = df[[0, 1]][Y == 0]
    X1 = df[[0, 1]][Y == 1]
    sigma0 = np.dot(X0.T, X0) / len(X0)
    sigma1 = np.dot(X1.T, X1) / len(X1)
    
    # print 'means: {}, {}'.format(mu0, mu1)
    # print 'covariances: {}, {}'.format(sigma0, sigma1)
    
    # Decision boudary
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    alpha = -0.5 * (sigma0_inv - sigma1_inv)
    beta = np.dot(sigma1_inv, mu1) - np.dot(sigma0_inv, mu0)
    gamma = math.log(pi / (1 - pi)) - 0.5 * (np.dot(np.dot(mu1.T, sigma1_inv), mu1) - np.dot(np.dot(mu0.T, sigma0_inv), mu0))
    return alpha, beta, gamma

def evaluate(x, coefs):
    A, b, gamma = coefs
    return np.dot(np.dot(x.T, A), x) + np.dot(b.T, x) + gamma > 0

def evaluate_qda(df, coeffs):
    l = []
    for x in df[[0,1]].values:
        l.append(evaluate(x, coeffs))
    return np.array(l)
    
def misscl_qda(df, coeffs):
    l = evaluate_qda(df, coeffs)
    return 1 - (df[2] == np.array(l)).mean()


# In[12]:

def plot_qda(df, title=None):
    ax = df.loc[df[2] == 1].plot(kind='scatter', x=0, y=1, color='blue', label='Gaussian 1')
    df.loc[df[2] == 0].plot(kind='scatter', x=0, y=1, ax=ax, color='green', label='Gaussian 2')

    # Taken from sklearn doc: http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
    coeffs = qda(df)
    nx, ny = 200, 200
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = evaluate_qda(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]), coeffs)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, [0], linewidths=.2)
    
    if title:
        plt.title(title)
    plt.show()
    
plot_qda(classA, 'Classification with QDA on classificationA')
plot_qda(classB, 'Classification with QDA on classificationB')
plot_qda(classC, 'Classification with QDA on classificationC')


# (d). Misclassification error for QDA

# In[13]:

print 'Missclassification error with QDA classifier\n'

for klass, data in classes.iteritems():
    weights = qda(data)
    error_train = misscl_qda(data, weights)
    error_test = misscl_qda(test[klass], weights)

    print '{} on the train data and {} on the test data for QDA on {}'.format(
            error_train, error_test, klass)


# The QDA classifier has poor results compared to the other classifers, it's probably because of the fact that even if we added a freedom degree compared to the LDA classifier (we don't assume the two gaussian have the same covariance matrix), our limited quantity of data makes it difficult to estimate this new parameters (bias-variance tradeoff)
