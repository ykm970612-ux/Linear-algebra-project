from Matrix import Matrix


def assert_raises(error_type, func):
    try:
        func()
        raise AssertionError(f"{error_type.__name__}가 발생해야 하는데 발생하지 않았습니다.")
    except error_type:
        pass


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

    


def test_null_space_basis():
    def assert_column_vector_close(vector, expected, eps=1e-9):
        assert vector.rows == len(expected)
        assert vector.cols == 1

        for i, value in enumerate(expected):
            assert abs(vector.data[i][0] - value) < eps

    def assert_is_null_vector(A, vector, eps=1e-9):
        result = A.multiply(vector)

        assert result.cols == 1
        for i in range(result.rows):
            assert abs(result.data[i][0]) < eps

    # 1. 자유변수가 1개인 경우
    A = Matrix([
        [1, 2],
        [2, 4]
    ])

    basis = A.null_space_basis()

    assert len(basis) == 1
    assert_column_vector_close(basis[0], [-2, 1])
    assert_is_null_vector(A, basis[0])
    assert len(basis) == A.cols - A.rank()

    # 2. 자유변수가 2개인 경우
    B = Matrix([
        [1, 2, 3],
        [2, 4, 6]
    ])

    basis = B.null_space_basis()

    assert len(basis) == 2
    assert_column_vector_close(basis[0], [-2, 1, 0])
    assert_column_vector_close(basis[1], [-3, 0, 1])

    for vector in basis:
        assert_is_null_vector(B, vector)

    assert len(basis) == B.cols - B.rank()

    # 3. 가역행렬: 영공간이 {0}뿐이므로 기저는 빈 리스트
    C = Matrix([
        [1, 0],
        [0, 1]
    ])

    basis = C.null_space_basis()

    assert basis == []
    assert len(basis) == C.cols - C.rank()

    # 4. 영행렬: 모든 변수가 자유변수
    D = Matrix([
        [0, 0],
        [0, 0]
    ])

    basis = D.null_space_basis()

    assert len(basis) == 2
    assert_column_vector_close(basis[0], [1, 0])
    assert_column_vector_close(basis[1], [0, 1])

    for vector in basis:
        assert_is_null_vector(D, vector)

    assert len(basis) == D.cols - D.rank()
    
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