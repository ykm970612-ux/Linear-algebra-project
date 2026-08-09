from Matrix import Matrix

def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass


def test_basic_operations():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    B = Matrix([
        [5, 6],
        [7, 8]
    ])

    assert A + B == Matrix([
        [6, 8],
        [10, 12]
    ])

    assert B - A == Matrix([
        [4, 4],
        [4, 4]
    ])

    assert A * 2 == Matrix([
        [2, 4],
        [6, 8]
    ])

    assert 2 * A == Matrix([
        [2, 4],
        [6, 8]
    ])

    print("Basic operation tests passed.")

def test_creation_methods():
    assert Matrix.identity(3) == Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    assert Matrix.zeros(2, 3) == Matrix([
        [0, 0, 0],
        [0, 0, 0]
    ])

    assert Matrix.diagonal([1, 2, 3]) == Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])

    print("Creation method tests passed.")

def test_transpose_and_multiply():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    B = Matrix([
        [5, 6],
        [7, 8]
    ])

    assert A.transpose() == Matrix([
        [1, 3],
        [2, 4]
    ])

    assert A @ B == Matrix([
        [19, 22],
        [43, 50]
    ])

    C = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert C.transpose() == Matrix([
        [1, 4],
        [2, 5],
        [3, 6]
    ])

    print("Transpose and multiply tests passed.")