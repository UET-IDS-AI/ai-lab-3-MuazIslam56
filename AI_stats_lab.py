# Lab3.py

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------
# Question 1: Linear Regression Pipeline
# -------------------------
def diabetes_linear_pipeline():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Top 3 features by absolute coefficient
    coef = model.coef_
    top3_idx = np.argsort(np.abs(coef))[-3:]
    top3_features = [diabetes.feature_names[i] for i in top3_idx]

    return train_mse, test_mse, train_r2, test_r2, top3_features


# -------------------------
# Question 2: Cross-Validation – Linear Regression
# -------------------------
def diabetes_cross_validation():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    return mean_r2, std_r2


# -------------------------
# Question 3: Logistic Regression Pipeline
# -------------------------
def cancer_logistic_pipeline():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    return train_acc, test_acc, precision, recall, f1


# -------------------------
# Question 4: Regularization in Logistic Regression
# -------------------------
def cancer_logistic_regularization():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=5000)
        model.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        results[C] = (train_acc, test_acc)

    return results


# -------------------------
# Question 5: Cross-Validation – Logistic Regression
# -------------------------
def cancer_cross_validation():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    return mean_acc, std_acc
