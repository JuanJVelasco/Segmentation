from ..data.loader import load_dataset

def test_shapes():
    X_train, X_val, y_train, y_val = load_dataset()
    assert X_train.shape[1:] == (256, 256, 3)
    assert y_train.shape[1:] == (256, 256, 1)
