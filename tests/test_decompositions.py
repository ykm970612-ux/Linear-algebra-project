from Matrix import Matrix


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass


def test_lu_decomposition():
    A = Matrix([
        [2, 3],
        [4, 7]
    ])

    L, U = A.lu_decomposition()

    assert L.is_lower_triangular()
    assert U.is_upper_triangular()
    assert L @ U == A


def test_plu_decomposition():
    test_cases = [
        # 행교환이 필요 없는 경우
        (
            [
                [4, 3],
                [2, 1]
            ],
            [
                [1, 0],
                [0, 1]
            ]
        ),

        # 첫 번째 단계에서 행교환
        (
            [
                [0, 2],
                [1, 3]
            ],
            [
                [0, 1],
                [1, 0]
            ]
        ),

        # 소거 이후 단계에서 행교환
        (
            [
                [4, 1, 1],
                [2, 0, 1],
                [1, 3, 1]
            ],
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]
        )
    ]

    for A_data, expected_P_data in test_cases:
        A = Matrix(A_data)
        original = Matrix(A_data)
        expected_P = Matrix(expected_P_data)

        P, L, U = A.plu_decomposition()

        # 예상한 행교환인지 검사
        assert P == expected_P

        # PLU의 핵심 관계
        assert P @ A == L @ U

        # L과 U의 구조
        assert L.is_lower_triangular()
        assert U.is_upper_triangular()

        # L의 대각 원소는 모두 1
        for i in range(L.rows):
            assert abs(L.data[i][i] - 1) < Matrix.EPS

        # P가 순열행렬인지 검사
        assert P @ P.transpose() == Matrix.identity(P.rows)

        # 분해 과정에서 원본이 변경되지 않았는지 검사
        assert A == original

    # 특이행렬 검사
    singular_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    assert_raises(
        ValueError,
        lambda: singular_A.plu_decomposition()

    )

def test_qr_decomposition():
    A = Matrix([
        [1, 1],
        [1, 0],
        [0, 1]
    ])

    Q, R = A.qr_decomposition()

    assert Q.is_orthonormal()
    assert R.is_upper_triangular()
    assert Q @ R == A

    B = Matrix([
        [1, 2],
        [3, 4]
    ])

    Q_B, R_B = B.qr_decomposition()

    assert Q_B.is_orthonormal()
    assert R_B.is_upper_triangular()
    assert Q_B @ R_B == B

    C = Matrix([
        [1, 2],
        [2, 4]
    ])

    assert_raises(ValueError, lambda: C.qr_decomposition())

    print("QR decomposition tests passed.")
    

