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

    

    print("\nall tests passed")


if __name__ == "__main__":
    run_all_tests()