from Matrix import Matrix


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass


def test_solve_general():
    def assert_column_vector_close(vector, expected, eps=1e-9):
        assert vector.rows == len(expected)
        assert vector.cols == 1

        for i, value in enumerate(expected):
            assert abs(vector.data[i][0] - value) < eps

    def assert_matrix_product(A, x, expected, eps=1e-9):
        result = A.multiply(x)
        assert_column_vector_close(result, expected, eps)

    # 1. 무한히 많은 해
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6]
    ])

    b = Matrix([
        [4],
        [8]
    ])

    particular, basis = A.solve_general(b)

    # 자유변수를 모두 0으로 둔 특정해
    assert_column_vector_close(particular, [4, 0, 0])
    assert_matrix_product(A, particular, [4, 8])

    # 영공간 기저
    assert len(basis) == 2
    assert_column_vector_close(basis[0], [-2, 1, 0])
    assert_column_vector_close(basis[1], [-3, 0, 1])

    for vector in basis:
        assert_matrix_product(A, vector, [0, 0])

    # 2. 유일해
    A = Matrix([
        [2, 1],
        [1, -1]
    ])

    b = Matrix([
        [5],
        [1]
    ])

    particular, basis = A.solve_general(b)

    assert_column_vector_close(particular, [2, 1])
    assert_matrix_product(A, particular, [5, 1])
    assert basis == []

    # 3. 해가 없는 경우
    A = Matrix([
        [1, 1],
        [2, 2]
    ])

    b = Matrix([
        [1],
        [3]
    ])

    assert_raises(
        ValueError,
        lambda: A.solve_general(b)
    )

    # 4. b가 Matrix가 아닌 경우
    A = Matrix([
        [1, 0],
        [0, 1]
    ])

    assert_raises(
        TypeError,
        lambda: A.solve_general([1, 2])
    )

    # 5. b가 열벡터가 아닌 경우
    row_b = Matrix([
        [1, 2]
    ])

    assert_raises(
        ValueError,
        lambda: A.solve_general(row_b)
    )

    # 6. A와 b의 행 개수가 다른 경우
    wrong_size_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(
        ValueError,
        lambda: A.solve_general(wrong_size_b)
    )
def test_least_squares_qr():
    def assert_column_vector_close(vector, expected, eps=1e-8):
        assert vector.rows == len(expected)
        assert vector.cols == 1

        for i, value in enumerate(expected):
            assert abs(vector.data[i][0] - value) < eps

    def assert_zero_vector(vector, eps=1e-8):
        assert vector.cols == 1

        for i in range(vector.rows):
            assert abs(vector.data[i][0]) < eps

    # 1. 정확한 해가 존재하는 tall 행렬
    A = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    b = Matrix([
        [2],
        [3],
        [5]
    ])

    x_hat = A.least_squares_qr(b)

    assert_column_vector_close(x_hat, [2, 3])
    assert_column_vector_close(A.multiply(x_hat), [2, 3, 5])

    # 2. 정확한 해가 없는 과잉결정 시스템
    A = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    b = Matrix([
        [1],
        [2],
        [2]
    ])

    x_hat = A.least_squares_qr(b)

    assert_column_vector_close(x_hat, [2 / 3, 5 / 3])

    # 최소제곱해의 잔차는 A의 모든 열에 직교해야 한다.
    prediction = A.multiply(x_hat)
    residual = b.subtract(prediction)
    orthogonality = A.transpose().multiply(residual)

    assert_zero_vector(orthogonality)

    # 3. 정사각행렬에서는 일반적인 정확한 해와 같아야 한다.
    A = Matrix([
        [2, 1],
        [1, -1]
    ])

    b = Matrix([
        [5],
        [1]
    ])

    x_hat = A.least_squares_qr(b)

    assert_column_vector_close(x_hat, [2, 1])
    assert_column_vector_close(A.multiply(x_hat), [5, 1])

    # 4. b가 Matrix가 아닌 경우
    assert_raises(
        TypeError,
        lambda: A.least_squares_qr([5, 1])
    )

    # 5. b가 열벡터가 아닌 경우
    row_b = Matrix([
        [5, 1]
    ])

    assert_raises(
        ValueError,
        lambda: A.least_squares_qr(row_b)
    )

    # 6. A와 b의 행 개수가 다른 경우
    wrong_size_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(
        ValueError,
        lambda: A.least_squares_qr(wrong_size_b)
    )

    # 7. wide 행렬은 현재 QR 최소제곱 구현에서 지원하지 않는다.
    wide_A = Matrix([
        [1, 0, 1],
        [0, 1, 1]
    ])

    wide_b = Matrix([
        [1],
        [2]
    ])

    assert_raises(
        ValueError,
        lambda: wide_A.least_squares_qr(wide_b)
    )

    # 8. 열종속 행렬은 R이 가역이 아니므로 처리할 수 없다.
    dependent_A = Matrix([
        [1, 2],
        [2, 4],
        [3, 6]
    ])

    dependent_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(
        ValueError,
        lambda: dependent_A.least_squares_qr(dependent_b)
    )