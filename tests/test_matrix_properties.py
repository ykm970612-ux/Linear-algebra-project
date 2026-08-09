from Matrix import Matrix


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass



def test_determinant_inverse_rank():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    assert abs(A.determinant() - (-2)) < 1e-10

    assert A.inverse() == Matrix([
        [-2, 1],
        [1.5, -0.5]
    ])

    assert A @ A.inverse() == Matrix.identity(2)

    B = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ])

    assert B.rank() == 2

    print("Determinant, inverse, rank tests passed.")


def test_row_echelon():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ])

    ref = A.row_echelon()

    assert ref.rank() == 2

    print("Row echelon tests passed.")


def test_rref():
    A = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = Matrix([
        [1, 0, -1],
        [0, 1, 2]
    ])

    assert A.rref() == expected

    B = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])

    expected_B = Matrix([
        [1, 2, 3],
        [0, 0, 0],
        [0, 0, 0]
    ])

    assert B.rref() == expected_B

    print("RREF tests passed.")