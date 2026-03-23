import numpy as np


def test_mlx_import():
    import mlx.core as mx

    x = mx.ones((3, 3))
    assert x.shape == (3, 3)
    print(f"MLX array:\n{x}")


def test_mlx_operations():
    import mlx.core as mx

    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0]


def test_mlx_nn():
    import mlx.core as mx
    import mlx.nn as nn

    layer = nn.Linear(16, 8)
    x = mx.random.normal((2, 16))
    y = layer(x)
    assert y.shape == (2, 8)


def test_numpy():
    x = np.random.randn(5, 5)
    assert x.shape == (5, 5)
    print(f"NumPy version: {np.__version__}")


def test_train_imports():
    """Verify train.py's key imports resolve."""
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer

    assert isinstance(MAX_SEQ_LEN, int)
    assert isinstance(TIME_BUDGET, (int, float))
    print(f"MAX_SEQ_LEN={MAX_SEQ_LEN}, TIME_BUDGET={TIME_BUDGET}")
