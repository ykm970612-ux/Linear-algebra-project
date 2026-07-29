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


def test_augment():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    expected = Matrix([
        [1, 2, 5],
        [3, 4, 11]
    ])

    assert A.augment(b) == expected

    C = Matrix([
        [1, 2, 3]
    ])

    assert_raises(ValueError, lambda: A.augment(C))

    print("Augment tests passed.")


def test_solve_unique():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    expected = Matrix([
        [1],
        [2]
    ])

    assert A.solve_unique(b) == expected

    tall_A = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    tall_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert tall_A.solve_unique(tall_b) == Matrix([
        [1],
        [2]
    ])

    no_solution_A = Matrix([
        [1, 1],
        [2, 2]
    ])

    no_solution_b = Matrix([
        [3],
        [7]
    ])

    assert_raises(ValueError, lambda: no_solution_A.solve_unique(no_solution_b))

    infinite_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    infinite_b = Matrix([
        [3],
        [6]
    ])

    assert_raises(ValueError, lambda: infinite_A.solve_unique(infinite_b))

    print("Solve unique tests passed.")


def test_analyze_system():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    status, rref_augmented = A.analyze_system(b)
    assert status == "unique"

    no_solution_A = Matrix([
        [1, 1],
        [2, 2]
    ])

    no_solution_b = Matrix([
        [3],
        [7]
    ])

    status, rref_augmented = no_solution_A.analyze_system(no_solution_b)
    assert status == "no solution"

    infinite_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    infinite_b = Matrix([
        [3],
        [6]
    ])

    status, rref_augmented = infinite_A.analyze_system(infinite_b)
    assert status == "infinite solutions"

    print("Analyze system tests passed.")


def test_has_solution():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    assert A.has_solution(b) == True

    no_solution_A = Matrix([
        [1, 1],
        [2, 2]
    ])

    no_solution_b = Matrix([
        [3],
        [7]
    ])

    assert no_solution_A.has_solution(no_solution_b) == False

    infinite_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    infinite_b = Matrix([
        [3],
        [6]
    ])

    assert infinite_A.has_solution(infinite_b) == True

    print("Has solution tests passed.")


def test_least_squares():
    A = Matrix([
        [1, 1],
        [1, 2],
        [1, 3]
    ])

    b = Matrix([
        [1],
        [2],
        [2]
    ])

    expected = Matrix([
        [2 / 3],
        [1 / 2]
    ])

    assert A.least_squares(b) == expected

    singular_A = Matrix([
        [1, 2],
        [2, 4],
        [3, 6]
    ])

    singular_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(ValueError, lambda: singular_A.least_squares(singular_b))

    print("Least squares tests passed.")


def test_least_squares_utilities():
    A = Matrix([
        [1, 1],
        [1, 2],
        [1, 3]
    ])

    b = Matrix([
        [1],
        [2],
        [2]
    ])

    expected_prediction = Matrix([
        [7 / 6],
        [5 / 3],
        [13 / 6]
    ])

    assert A.least_squares_prediction(b) == expected_prediction

    expected_residual = Matrix([
        [1 - 7 / 6],
        [2 - 5 / 3],
        [2 - 13 / 6]
    ])

    assert A.residual(b) == expected_residual

    assert abs(A.residual_norm(b) - expected_residual.frobenius_norm()) < 1e-10

    print("Least squares utility tests passed.")


def test_frobenius_norm():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    assert abs(A.frobenius_norm() - (30 ** 0.5)) < 1e-10

    v = Matrix([
        [3],
        [4]
    ])

    assert abs(v.frobenius_norm() - 5) < 1e-10

    print("Frobenius norm tests passed.")


def test_dot_and_normalize():
    u = Matrix([
        [1],
        [2],
        [3]
    ])

    v = Matrix([
        [4],
        [5],
        [6]
    ])

    assert u.dot(v) == 32

    row_u = Matrix([
        [1, 2, 3]
    ])

    assert row_u.dot(v) == 32

    normalized = Matrix([
        [3],
        [4]
    ]).normalize()

    assert normalized == Matrix([
        [3 / 5],
        [4 / 5]
    ])

    zero_v = Matrix([
        [0],
        [0]
    ])

    assert_raises(ValueError, lambda: zero_v.normalize())

    print("Dot and normalize tests passed.")


def test_get_column_and_from_columns():
    A = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    c1 = Matrix([
        [1],
        [3],
        [5]
    ])

    c2 = Matrix([
        [2],
        [4],
        [6]
    ])

    assert A.get_column(0) == c1
    assert A.get_column(1) == c2

    assert Matrix.from_columns([c1, c2]) == A

    assert_raises(IndexError, lambda: A.get_column(2))
    assert_raises(IndexError, lambda: A.get_column(-1))
    assert_raises(TypeError, lambda: A.get_column(1.5))

    print("Get column and from columns tests passed.")


def test_gram_schmidt():
    A = Matrix([
        [1, 0],
        [0, 1]
    ])

    assert A.gram_schmidt() == A

    B = Matrix([
        [1, 1],
        [1, 0],
        [0, 1]
    ])

    Q = B.gram_schmidt()

    q1 = Q.get_column(0)
    q2 = Q.get_column(1)

    assert abs(q1.dot(q2)) < 1e-10
    assert abs(q1.frobenius_norm() - 1) < 1e-10
    assert abs(q2.frobenius_norm() - 1) < 1e-10

    C = Matrix([
        [1, 2],
        [2, 4]
    ])

    assert_raises(ValueError, lambda: C.gram_schmidt())

    print("Gram-Schmidt tests passed.")


def test_is_full_column_rank():
    A = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    assert A.is_full_column_rank() == True

    B = Matrix([
        [1, 2],
        [3, 4]
    ])

    assert B.is_full_column_rank() == True

    C = Matrix([
        [1, 2],
        [2, 4],
        [3, 6]
    ])

    assert C.is_full_column_rank() == False

    D = Matrix([
        [1, 0, 2],
        [0, 1, 3]
    ])

    assert D.is_full_column_rank() == False

    print("Full column rank tests passed.")


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


def test_is_upper_triangular():
    A = Matrix([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ])

    assert A.is_upper_triangular() == True

    B = Matrix([
        [1, 2, 3],
        [0, 4, 5],
        [1, 0, 6]
    ])

    assert B.is_upper_triangular() == False

    C = Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])

    assert C.is_upper_triangular() == True

    E = Matrix([
        [1, 2, 3],
        [0, 4, 5]
    ])

    assert_raises(ValueError, lambda: E.is_upper_triangular())

    print("Upper triangular tests passed.")


def test_is_orthonormal():
    I = Matrix.identity(3)
    assert I.is_orthonormal() == True

    A = Matrix([
        [1, 0],
        [0, 1],
        [0, 0]
    ])

    assert A.is_orthonormal() == True

    B = Matrix([
        [2, 0],
        [0, 1]
    ])

    assert B.is_orthonormal() == False

    C = Matrix([
        [1, 1],
        [0, 1]
    ])

    assert C.is_orthonormal() == False

    print("Orthonormal tests passed.")


def test_errors():
    assert_raises(ValueError, lambda: Matrix([]))
    assert_raises(ValueError, lambda: Matrix([[1, 2], [3]]))
    assert_raises(TypeError, lambda: Matrix([[1, "a"]]))
    assert_raises(TypeError, lambda: Matrix([[True, 1]]))

    assert_raises(ValueError, lambda: Matrix.identity(0))
    assert_raises(ValueError, lambda: Matrix.zeros(0, 3))
    assert_raises(ValueError, lambda: Matrix.diagonal([]))

    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    C = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert_raises(ValueError, lambda: A + C)
    assert_raises(ValueError, lambda: C.determinant())

    print("Error tests passed.")

def test_lu_decomposition():
    A = Matrix([
        [2, 3],
        [4, 7]
    ])

    L, U = A.lu_decomposition()

    assert L.is_lower_triangular()
    assert U.is_upper_triangular()
    assert L @ U == A

def test_forward_substitution():
    L = Matrix([
        [1, 0],
        [2, 1]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    expected = Matrix([
        [5],
        [1]
    ])

    assert L.forward_substitution(b) == expected

    print("Forward substitution tests passed.")

def test_back_substitution():
    U = Matrix([
        [2, 3],
        [0, 1]
    ])

    b = Matrix([
        [5],
        [1]
    ])

    expected = Matrix([
        [1],
        [1]
    ])

    assert U.back_substitution(b) == expected

    print("Back substitution tests passed.")

def test_solve_lu():
    A = Matrix([
        [2, 3],
        [4, 7]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    expected = Matrix([
        [1],
        [1]
    ])

    assert A.solve_lu(b) == expected

    print("Solve LU tests passed.")

def test_dot_rejects_non_vector():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    v = Matrix([
       [1],
       [2]
    ])

    assert_raises(ValueError, lambda: A.dot(v))
    assert_raises(ValueError,lambda :v.dot(A))

def test_constructor_copies_input_data():
    data = [
        [1, 2],
        [3, 4]
    ]
    A = Matrix(data)
    data[0][0] = 999
    assert A.data[0][0] == 1


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

def test_solve_plu():
    test_cases = [
        # 행교환이 필요 없는 경우
        (
            [
                [4, 3],
                [2, 1]
            ],
            [
                [1],
                [2]
            ]
        ),

        # 첫 pivot에서 행교환이 필요한 경우
        (
            [
                [0, 2],
                [1, 3]
            ],
            [
                [1],
                [2]
            ]
        ),

        # 두 번째 pivot에서 행교환이 필요한 경우
        (
            [
                [4, 1, 1],
                [2, 0, 1],
                [1, 3, 1]
            ],
            [
                [1],
                [2],
                [3]
            ]
        )
    ]

    for A_data, expected_x_data in test_cases:
        A = Matrix(A_data)
        expected_x = Matrix(expected_x_data)

        # 알고 있는 해를 이용해 b 생성
        b = A @ expected_x

        original_A = Matrix(A_data)
        original_b = Matrix(b.data)

        result = A.solve_plu(b)

        # 예상한 해인지 검사
        assert result == expected_x

        # 실제로 Ax=b를 만족하는지 검사
        assert A @ result == b

        # RREF 기반 풀이와 같은지 검사
        assert result == A.solve_unique(b)

        # 원본 입력이 변경되지 않았는지 검사
        assert A == original_A
        assert b == original_b

    # 특이행렬
    singular_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    singular_b = Matrix([
        [3],
        [6]
    ])

    assert_raises(
        ValueError,
        lambda: singular_A.solve_plu(singular_b)
    )

    # b가 열벡터가 아닌 경우
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    row_b = Matrix([
        [1, 2]
    ])

    assert_raises(
        ValueError,
        lambda: A.solve_plu(row_b)
    )

    # A와 b의 행 개수가 다른 경우
    wrong_size_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(
        ValueError,
        lambda: A.solve_plu(wrong_size_b)
    )

    # b가 Matrix가 아닌 경우
    assert_raises(
        TypeError,
        lambda: A.solve_plu([1, 2])
    )

    






