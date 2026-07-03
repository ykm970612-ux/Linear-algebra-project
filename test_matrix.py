from Matrix import Matrix


def test_basic_operations():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    assert A.shape() == (2, 2)
    assert A.is_square() is True

    assert A + B == Matrix([[6, 8], [10, 12]])
    assert A - B == Matrix([[-4, -4], [-4, -4]])

    assert A * 2 == Matrix([[2, 4], [6, 8]])
    assert 2 * A == Matrix([[2, 4], [6, 8]])

    assert A @ B == Matrix([[19, 22], [43, 50]])

    print("basic operations passed")


def test_creation_methods():
    Z = Matrix.zeros(2, 3)
    I = Matrix.identity(3)
    D = Matrix.diagonal([2, 5, 7])

    assert Z == Matrix([[0, 0, 0], [0, 0, 0]])

    assert I == Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    assert D == Matrix([
        [2, 0, 0],
        [0, 5, 0],
        [0, 0, 7]
    ])

    print("creation methods passed")


def test_transpose_and_multiply():
    C = Matrix([[1, 2, 3], [4, 5, 6]])
    D = Matrix([[1, 2], [3, 4], [5, 6]])

    assert C.shape() == (2, 3)

    assert C.transpose() == Matrix([
        [1, 4],
        [2, 5],
        [3, 6]
    ])

    assert C @ D == Matrix([
        [22, 28],
        [49, 64]
    ])

    print("transpose and multiply passed")


def test_determinant_inverse_rank():
    A = Matrix([[1, 2], [3, 4]])

    assert A.determinant() == -2
    assert A.rank() == 2

    expected_inverse = Matrix([
        [-2.0, 1.0],
        [1.5, -0.5]
    ])

    assert A.inverse() == expected_inverse

    # 역행렬 검증: A * A^-1 = I
    assert A @ A.inverse() == Matrix.identity(2)

    print("determinant, inverse, rank passed")


def test_row_echelon():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 7],
        [1, 1, 1]
    ])

    expected = Matrix([
        [1, 2, 3],
        [0.0, -1.0, -2.0],
        [0.0, 0.0, 1.0]
    ])

    assert A.row_echelon() == expected
    assert A.rank() == 3

    print("row echelon passed")

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


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass


def test_errors():
    A = Matrix([[1, 2], [3, 4]])
    C = Matrix([[1, 2, 3], [4, 5, 6]])

    assert_raises(ValueError, lambda: Matrix([]))
    assert_raises(ValueError, lambda: Matrix([[1, 2], [3]]))
    assert_raises(TypeError, lambda: Matrix([[1, "a"]]))
    assert_raises(TypeError, lambda: Matrix([[True, 1]]))

    assert_raises(ValueError, lambda: Matrix.identity(0))
    assert_raises(ValueError, lambda: Matrix.zeros(0, 3))

    assert_raises(ValueError, lambda: A + C)
    assert_raises(ValueError, lambda: C.determinant())

    print("error tests passed")

def test_sigular_rank():
    S = Matrix([[1, 2], [2, 4]])
    R = Matrix([[1, 2, 3], [2, 4, 6]])

    assert R.rank() == 1
    assert S.determinant() == 0
    assert S.rank() == 1

    print("singula and rank passed")

def test_solve_unique():
    # 1. 정사각행렬: 유일해 존재
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


    # 2. 다른 정사각행렬: 유일해 존재
    B = Matrix([
        [2, 1],
        [5, 3]
    ])

    c = Matrix([
        [1],
        [2]
    ])

    expected_c = Matrix([
        [1],
        [-1]
    ])

    assert B.solve_unique(c) == expected_c


    # 3. tall matrix: 방정식은 3개, 변수는 2개지만 유일해 존재
    # x = 1, y = 2를 만족하는 시스템
    C = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    d = Matrix([
        [1],
        [2],
        [3]
    ])

    expected_d = Matrix([
        [1],
        [2]
    ])

    assert C.solve_unique(d) == expected_d


    # 4. 해 없음: 0 = 1 같은 모순 발생
    no_solution_A = Matrix([
        [1, 1],
        [2, 2]
    ])

    no_solution_b = Matrix([
        [3],
        [7]
    ])

    assert_raises(ValueError, lambda: no_solution_A.solve_unique(no_solution_b))


    # 5. 무한히 많은 해: pivot이 모든 변수 열에 없음
    infinite_A = Matrix([
        [1, 2],
        [2, 4]
    ])

    infinite_b = Matrix([
        [3],
        [6]
    ])

    assert_raises(ValueError, lambda: infinite_A.solve_unique(infinite_b))


    # 6. b가 열벡터가 아님
    wrong_b = Matrix([
        [1, 2]
    ])

    assert_raises(ValueError, lambda: A.solve_unique(wrong_b))


    # 7. b의 행 개수가 A와 다름
    wrong_rows_b = Matrix([
        [1],
        [2],
        [3]
    ])

    assert_raises(ValueError, lambda: A.solve_unique(wrong_rows_b))


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
    # 유일해가 있으므로 True
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    b = Matrix([
        [5],
        [11]
    ])

    assert A.has_solution(b) == True


    # 해가 없으므로 False
    no_solution_A = Matrix([
        [1, 1],
        [2, 2]
    ])

    no_solution_b = Matrix([
        [3],
        [7]
    ])

    assert no_solution_A.has_solution(no_solution_b) == False


    # 무한히 많은 해가 있으므로 True
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
    # 1. 정확한 해가 없는 tall matrix
    # y ≈ c + mx 형태의 직선 fitting
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

    # 최소제곱해:
    # c = 2/3, m = 1/2
    expected = Matrix([
        [2 / 3],
        [1 / 2]
    ])

    assert A.least_squares(b) == expected


    # 2. 정확한 해가 있는 경우에도 least squares는 같은 해를 반환해야 함
    B = Matrix([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    c = Matrix([
        [1],
        [2],
        [3]
    ])

    expected_c = Matrix([
        [1],
        [2]
    ])

    assert B.least_squares(c) == expected_c


    # 3. b가 열벡터가 아닌 경우
    wrong_b = Matrix([
        [1, 2, 3]
    ])

    assert_raises(ValueError, lambda: A.least_squares(wrong_b))


    # 4. b의 행 개수가 A와 다른 경우
    wrong_rows_b = Matrix([
        [1],
        [2]
    ])

    assert_raises(ValueError, lambda: A.least_squares(wrong_rows_b))


    # 5. A의 열들이 선형종속이면 A^T A가 singular라서 유일한 least squares 해를 못 구함
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

def test_projection_and_residual():
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

    # least squares 해:
    # x_hat = [2/3, 1/2]
    #
    # projection = A x_hat
    # row1: 2/3 + 1/2 = 7/6
    # row2: 2/3 + 1   = 5/3
    # row3: 2/3 + 3/2 = 13/6
    expected_projection = Matrix([
        [7 / 6],
        [5 / 3],
        [13 / 6]
    ])

    assert A.project_column_space(b) == expected_projection

    # residual = b - projection
    expected_residual = Matrix([
        [1 - 7 / 6],
        [2 - 5 / 3],
        [2 - 13 / 6]
    ])

    assert A.residual(b) == expected_residual

    print("Projection and residual tests passed.")


def test_dot():
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


    # 행벡터끼리도 가능해야 함
    row_u = Matrix([
        [1, 2, 3]
    ])

    row_v = Matrix([
        [4, 5, 6]
    ])

    assert row_u.dot(row_v) == 32


    # 행벡터와 열벡터도 원소 개수가 같으면 가능
    assert row_u.dot(v) == 32


    # 길이가 다르면 에러
    short_v = Matrix([
        [1],
        [2]
    ])

    assert_raises(ValueError, lambda: u.dot(short_v))


    # 일반 행렬은 dot 불가능
    A = Matrix([
        [1, 2],
        [3, 4]
    ])

    assert_raises(ValueError, lambda: A.dot(u))

    print("Dot tests passed.")



def test_from_columns():
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

    expected = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    assert Matrix.from_columns([c1, c2]) == expected


    # 열벡터가 하나만 있어도 가능
    assert Matrix.from_columns([c1]) == c1


    # 빈 리스트는 불가능
    assert_raises(ValueError, lambda: Matrix.from_columns([]))


    # columns는 리스트여야 함
    assert_raises(TypeError, lambda: Matrix.from_columns(c1))


    # 각 원소는 Matrix여야 함
    assert_raises(TypeError, lambda: Matrix.from_columns([c1, [2, 4, 6]]))


    # 열벡터가 아니면 불가능
    row_vector = Matrix([
        [1, 2, 3]
    ])

    assert_raises(ValueError, lambda: Matrix.from_columns([c1, row_vector]))


    # 행 개수가 다르면 불가능
    short_col = Matrix([
        [1],
        [2]
    ])

    assert_raises(ValueError, lambda: Matrix.from_columns([c1, short_col]))

    print("From columns tests passed.")

def test_gram_schmidt():
    # 이미 직교정규인 경우
    A = Matrix([
        [1, 0],
        [0, 1]
    ])

    Q = A.gram_schmidt()

    expected = Matrix([
        [1, 0],
        [0, 1]
    ])

    assert Q == expected


    # 일반적인 선형독립 열벡터
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


    # 선형종속 열벡터는 Gram-Schmidt 불가능
    C = Matrix([
        [1, 2],
        [2, 4]
    ])

    assert_raises(ValueError, lambda: C.gram_schmidt())

    print("Gram-Schmidt tests passed.")

def test_qr_decomposition():
    A = Matrix([
        [1, 1],
        [1, 0],
        [0, 1]
    ])

    Q, R = A.qr_decomposition()

    q1 = Q.get_column(0)
    q2 = Q.get_column(1)

    # Q의 열벡터들은 서로 수직이어야 함
    assert abs(q1.dot(q2)) < 1e-10

    # Q의 열벡터들은 길이가 1이어야 함
    assert abs(q1.frobenius_norm() - 1) < 1e-10
    assert abs(q2.frobenius_norm() - 1) < 1e-10

    # A = Q @ R 이어야 함
    assert Q @ R == A


    # 선형종속 열벡터는 QR 분해 불가능
    B = Matrix([
        [1, 2],
        [2, 4]
    ])

    assert_raises(ValueError, lambda: B.qr_decomposition())

    print("QR decomposition tests passed.")



def run_all_tests():
    test_basic_operations()
    test_creation_methods()
    test_transpose_and_multiply()
    test_determinant_inverse_rank()
    test_row_echelon()
    test_sigular_rank()
    test_rref()
    test_solve_unique()
    test_errors()
    test_has_solution()
    test_analyze_system()
    test_least_squares()
    test_projection_and_residual()
    test_from_columns()
    test_dot()
    test_gram_schmidt()
    test_qr_decomposition()
    

    print("\nall tests passed")


if __name__ == "__main__":
    run_all_tests()