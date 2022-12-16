import numpy as np
import math
import random

def gradient(beta, X, Y):
    z = np.subtract(Y, 1)
    w = np.power(math.e, np.dot(-X, beta.T))
    p = np.divide(w, np.add(1, w))
    q = z + p

    delta = np.dot(q, X)

    return delta

##Log likelihood function. Returns singular value
def log_likelihood(beta, X, Y):
    
    t = np.dot(X, beta.T)
    w = np.subtract(Y, 1)
    w = np.dot(w, t)

    m = np.power(math.e, -t)
    m = np.add(1, m)
    z = np.log(m)
    z = np.sum(z)

    return w - z

def fit(X, Y, epsilon, step, start, end, beta=0, max=2000):
    X = X[start:end]
    Y = Y[start:end]
    data_points = X.shape[0]
    dimensions = X.shape[1]

    likelihood0 = []

    delta = 0
    itr = 0
    itrs = []
    #Intitialization of beta vector
    if beta == 0:
        beta = []
        for i in range(dimensions):
            beta.append(random.random() * 0)
        beta = np.array(beta)

    #Gradient ascent
    while True:
        
        grad = gradient(beta, X, Y) #Gradient

        beta = beta + (step*grad) #Recalculating beta
        delta = np.linalg.norm(grad) #Evaluating delta
        log = log_likelihood(beta, X, Y) #Log likelihood value for current iteration

        likelihood0.append(log)
        itr+=1
        itrs.append(itr)
        #Break conditions
        if delta < epsilon or itr > max:
            break
    
    return beta, likelihood0, itrs

def test(X, Y, beta, start, end):
    X = X[start:end]
    Y = Y[start:end]

    z = np.dot(X, beta)
    #Logistic Regression function for predicted y values
    pred_y = (1/(1+np.exp(-z)))
    pred_y = np.around(pred_y)
    correct = 0
    for i in range(len(pred_y)):
        if pred_y[i] == Y[start+i]:
            correct+=1
    
    return correct/len(pred_y)

def formatData(X, Y, v0, v1):
    lg = max(v0, v1)
    indices = []
    for i in range(len(Y)):
        if Y[i] == v0 or Y[i] == v1:
            indices.append(i)
    X = np.squeeze(np.take(X, indices, axis=0))
    Y = np.squeeze(np.floor(np.divide(np.take(Y, indices, axis=0), lg)))
    return X, Y

def accuracyEvaluation(correct, incorrect, total):
    correct = len(correct[0]), len(correct[1])
    incorrect = len(incorrect[0]), len(incorrect[1])
    print(f"Correct IMAGE1: {correct[0]}, correct IMAGE2: {correct[1]}")
    print(f"Incorrect IMAGE1: {incorrect[0]}, incorrect IMAGE2: {incorrect[1]}")
    acc = ((correct[0] + correct[1]) / total) * 100
    print(f"Accuracy: {acc}%")