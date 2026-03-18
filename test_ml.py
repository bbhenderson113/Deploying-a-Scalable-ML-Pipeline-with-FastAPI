import pytest
from ml.model import compute_model_metrics, inference, train_model

def test_model_metrics():
    """
    # This test checked the output of computer_model_metrics.
    # The test asserts that the outputs are floats and between 0 and 1.
    """
    y_test = [0, 1, 1]
    preds = [0, 1, 0]
     
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference_check():
    """
    # This test verifies that the output of interence (preds) contains the same number of elements as input (X_test)
    """
    X_train = [[0, 6], [1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 0]]
    y_train = [0, 0, 1, 1, 1, 0, 1]
    X_test = [[1, 2], [3, 4], [5, 6]]

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)


def test_train_model():
    """
    Tests to ensure that train_model returns a model capable of making predictions.
    """
    X_train = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    y_train = [0, 1, 0, 1, 0 ,1]

    model = train_model(X_train, y_train)
    preds = model.predict(X_train)

    assert model is not None
    assert hasattr(model, "predict")
    assert len(preds) == len(X_train)

