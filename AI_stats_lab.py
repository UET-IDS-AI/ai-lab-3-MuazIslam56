import Lab3 as A  

def test_diabetes_pipeline():
    train_mse, test_mse, train_r2, test_r2, top3 = A.diabetes_linear_pipeline()
    assert train_mse > 0, "Train MSE should be positive"
    assert -1 <= test_r2 <= 1, "Test R2 should be in [-1, 1]"
    assert len(top3) == 3, "Top 3 features should have length 3"

def test_diabetes_cv():
    mean_r2, std_r2 = A.diabetes_cross_validation()
    assert -1 <= mean_r2 <= 1, "Mean R2 should be in [-1, 1]"
    assert std_r2 >= 0, "Std of R2 should be non-negative"

def test_cancer_pipeline():
    train_acc, test_acc, precision, recall, f1, conf_matrix = A.cancer_logistic_pipeline()
    assert 0 <= train_acc <= 1, "Train accuracy should be between 0 and 1"
    assert 0 <= test_acc <= 1, "Test accuracy should be between 0 and 1"
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1-score should be between 0 and 1"
    assert conf_matrix.shape == (2,2), "Confusion matrix must be 2x2 for binary classification"

def test_logistic_regularization():
    results = A.cancer_logistic_regularization()
    assert len(results) == 5, "There should be 5 C values in results"

def test_logistic_cv():
    mean_acc, std_acc = A.cancer_cross_validation()
    assert 0 <= mean_acc <= 1, "Mean accuracy should be between 0 and 1"
    assert std_acc >= 0, "Std accuracy should be non-negative"
