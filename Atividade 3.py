#!/usr/bin/env python
# coding: utf-8

import json

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Classificadores
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Métricas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

# from sklearn.model_selection import cross_validate

bug0_filename = './datasets/bugs/linux/class0.txt'
bug1_filename = './datasets/bugs/linux/class1.txt'

def extract_descriptions(filename):
    bug_text = open(filename).read()
    bug_texts = bug_text.split('\n')
    return [ json.loads(bug)['DESCRIPTION'] for bug in bug_texts if len(bug) > 0  ]

# Carregamento dos dados
bug0_texts = extract_descriptions(bug0_filename)
bug1_texts = extract_descriptions(bug1_filename)

bugs_texts = bug0_texts + bug1_texts

# Array comprehension
y = [ 0 for bug in bug0_texts] + [ 1 for bug in bug1_texts]

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(bugs_texts)

############## Decision Tree ############## 
print('### Decision Tree ###')

clf_dt = DecisionTreeClassifier()

param_dict_dt = { 
    "criterion":['gini','entropy'],
    "max_depth":range(2,20)
}

grid_dt = GridSearchCV(
    clf_dt,
    param_grid = param_dict_dt,
    scoring = 'accuracy',
    cv=10)

grid_dt.fit(X, y)

print(f"Melhores valores  :")
print(f"- criterion       : {grid_dt.best_params_['criterion']}")
print(f"- max_depth       : {grid_dt.best_params_['max_depth']}" )
print(f"Acurácia          :  {grid_dt.best_score_}")

############## SVM Linear ############## 
print('### SVM Linear ###')

clf_svm_linear = LinearSVC()

param_dict_svm_linear = {
    'C': range(1,30)
}

grid_svm_linear = GridSearchCV(
    clf_svm_linear,
    param_grid = param_dict_svm_linear,
    scoring = 'accuracy',
    cv = 10)

grid_svm_linear.fit(X, y)

print(f"Melhores valores  :")
print(f"- C               :  {grid_svm_linear.best_params_['C']}")
print(f"Acurácia          :  {grid_svm_linear.best_score_}")

############## SVM ##############
print('### SVM ###')

clf_svm = svm.SVC()

param_dict_svm = [  
    {'C': [0.1,1,2,3,4,5,6,7,8,9,10,100,1000], 'kernel': ['linear']},
    {'C': [0.1,1,2,3,4,5,6,7,8,9,10,100,1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
] 

grid_svm = GridSearchCV(
    clf_svm,
    param_grid=param_dict_svm,
    scoring = 'accuracy',
    cv=10)

grid_svm.fit(X, y)

print(f"Melhores valores  :")
print(f"- C               :  {grid_svm.best_params_['C']}")
print(f"- kernel          :  {grid_svm.best_params_['kernel']}" )
print(f"Acurácia          :  {grid_svm.best_score_}")

############## Random Forest ##############
print('### Random Forest ###')

clf_rf = RandomForestClassifier()

param_dict_rf = {
    # número de árvores na floresta
    "n_estimators":range(2,30),
    # profundidade máxima da árvore
    "max_depth":range(2,10),
    # Os parâmetros abaixo não foram considerados devido ao fato de aumentar o tempo de processamento
    # número mínimo de amostras necessárias para dividir um nó interno
    # 'min_samples_split':range(2,10),
    # número mínimo de amostras necessárias para estar em um nó folha
    # 'min_samples_leaf':range(1,10)
}

grid_rf = GridSearchCV(
    clf_rf,
    param_grid=param_dict_rf,
    scoring = 'accuracy',
    n_jobs = 3,
    cv = 10)

grid_rf.fit(X, y)

print(f"Melhores valores  :")
print(f"- n_estimators    :  {grid_rf.best_params_['n_estimators']}")
print(f"- max_depth       :  {grid_rf.best_params_['max_depth']}" )
print(f"Acurácia          :  {grid_rf.best_score_}")

############## Regressão Logística ##############
print('### Regressão Logística ###')

clf_rl = LogisticRegression()

param_dict_rl = {
    "solver":  ['newton-cg','lbfgs','liblinear','sag'],
    "max_iter": range(95,106)
}
        
grid_rl = GridSearchCV(
    clf_rl,
    param_grid = param_dict_rl,
    scoring = 'accuracy',
    n_jobs = 3,
    cv = 10)

grid_rl.fit(X, y)

print(f"Melhores valores  :")
print(f"- solver          :  {grid_rl.best_params_['solver']}")
print(f"- max_ite         :  {grid_rl.best_params_['max_iter']}" )
print(f"Acurácia          :  {grid_rl.best_score_}")

############## Gaussian Naive Bayes ##############
print('### Gaussian Naive Bayes ###')

clf_nb = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

grid_nb = GridSearchCV(
    clf_nb,
    param_grid = params_NB,
    scoring = 'accuracy',
    cv = 10)

grid_nb.fit(X.toarray(), y)

print(f"Melhores valores  :")
print(f"- var_smoothing   : {grid_nb.best_params_['var_smoothing']}")
print(f"Acurácia          :  {grid_nb.best_score_}")



