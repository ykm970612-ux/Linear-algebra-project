from Matrix import Matrix


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass

def test_dominant_eigenpair():
    def assert_eigenpair(A, eigenvalue, eigenvector,
                         expected_eigenvalue, eps=1e-7):
        # 예상 지배 고유값과 일치하는지 확인
        assert abs(eigenvalue - expected_eigenvalue) < eps

        # 반환된 고유벡터가 열벡터인지 확인
        assert eigenvector.rows == A.rows
        assert eigenvector.cols == 1

        # 반환된 고유벡터가 길이 1로 정규화되었는지 확인
        assert abs(eigenvector.frobenius_norm() - 1) < eps

        # Av ≈ λv인지 잔차로 확인
        Av = A.multiply(eigenvector)
        lambda_v = eigenvector.scalar_multiply(eigenvalue)
        residual = Av.subtract(lambda_v)

        assert residual.frobenius_norm() < eps

    # 1. 양수 지배 고유값을 가진 대각행렬
    A = Matrix([
        [5, 0],
        [0, 2]
    ])

    initial_vector = Matrix([
        [1],
        [1]
    ])

    eigenvalue, eigenvector = A.dominant_eigenpair(
        initial_vector=initial_vector
    )

    assert_eigenpair(A, eigenvalue, eigenvector, 5)

    # 지배 고유벡터는 [1, 0]^T 방향이다.
    assert abs(abs(eigenvector.data[0][0]) - 1) < 1e-7
    assert abs(eigenvector.data[1][0]) < 1e-7

    # 2. 대각행렬이 아닌 대칭행렬
    B = Matrix([
        [2, 1],
        [1, 2]
    ])

    eigenvalue, eigenvector = B.dominant_eigenpair()

    assert_eigenpair(B, eigenvalue, eigenvector, 3)

    # 지배 고유벡터는 [1, 1]^T 방향이다.
    assert abs(
        eigenvector.data[0][0] - eigenvector.data[1][0]
    ) < 1e-7

    # 3. 음수 지배 고유값
    C = Matrix([
        [-5, 0],
        [0, 2]
    ])

    initial_vector = Matrix([
        [1],
        [1]
    ])

    eigenvalue, eigenvector = C.dominant_eigenpair(
        initial_vector=initial_vector
    )

    # 절대값이 가장 큰 고유값은 -5이다.
    assert_eigenpair(C, eigenvalue, eigenvector, -5)

    # 4. 비대칭행렬
    non_symmetric = Matrix([
        [1, 2],
        [0, 1]
    ])

    assert_raises(
        ValueError,
        lambda: non_symmetric.dominant_eigenpair()
    )

    # 5. initial_vector가 Matrix가 아닌 경우
    assert_raises(
        TypeError,
        lambda: A.dominant_eigenpair(
            initial_vector=[1, 1]
        )
    )

    # 6. 초기벡터가 열벡터가 아닌 경우
    row_vector = Matrix([
        [1, 1]
    ])

    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            initial_vector=row_vector
        )
    )

    # 7. 초기벡터의 행 개수가 잘못된 경우
    wrong_size_vector = Matrix([
        [1],
        [1],
        [1]
    ])

    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            initial_vector=wrong_size_vector
        )
    )

    # 8. 초기벡터가 영벡터인 경우
    zero_vector = Matrix([
        [0],
        [0]
    ])

    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            initial_vector=zero_vector
        )
    )

    # 9. 초기벡터가 A의 영공간에 들어간 경우
    singular_A = Matrix([
        [5, 0],
        [0, 0]
    ])

    null_vector = Matrix([
        [0],
        [1]
    ])

    assert_raises(
        ValueError,
        lambda: singular_A.dominant_eigenpair(
            initial_vector=null_vector
        )
    )

    # 10. 잘못된 최대 반복 횟수
    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            max_iterations=0
        )
    )

    assert_raises(
        TypeError,
        lambda: A.dominant_eigenpair(
            max_iterations=10.5
        )
    )

    # 11. 잘못된 tolerance
    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            tolerance=0
        )
    )

    assert_raises(
        ValueError,
        lambda: A.dominant_eigenpair(
            tolerance=-1e-10
        )
    )

    # 12. 제한된 반복 횟수 안에 수렴하지 못하는 경우
    assert_raises(
        RuntimeError,
        lambda: A.dominant_eigenpair(
            initial_vector=Matrix([[1], [1]]),
            max_iterations=1,
            tolerance=1e-15
        )
    )