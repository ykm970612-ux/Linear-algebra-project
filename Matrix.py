class Matrix:
    EPS = 1e-10

    def __init__(self,A):
        self._validate_matrix(A)


        self.data = A
        self.rows = len(A)
        self.cols = len(A[0])

    def _validate_matrix(self,A): # 행렬의 조건과 정확한 구조인지 확인한다.
        if not A: # 빈 리스트 인지 확인하여 잘못된 입력을 방지한다.
            raise ValueError("Empty Matrix")
        if not isinstance(A,list): # 구조가 리스트인지 확인.
            raise TypeError("Matrix data must be a list")

        if not all(isinstance(row,list) for row in A): #행의 구조가 리스트 인지 확인.
            raise TypeError("Each row must be a list")

        col_size = len(A[0])

        if col_size == 0: #열의 크기가 0인지 확인하여 빈 행렬인지 확인한다.
            raise ValueError("Matrix row cannot be empty")

        if not all(len(row) == col_size for row in A): #모든 행이 같은 크기인지 확인하여 행렬의 구조인지 확인한다..
            raise ValueError("All row must have the same length")
        
        if not all(isinstance(value,(int,float)) and not isinstance(value,bool)
                   for row in A
                   for value in row
        ): # 모든 요소가 숫자인지 확인하여 오류가 없게한다..
            raise TypeError("Matrix elments must be numbers")
        
    
    def _copy_data(self): #원본 행렬을 바꾸지 않기 위해 새로운 배열을 생성한다.

        rows = len(self.data)
        cols = len(self.data[0])
        C = [[0]*cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                C[i][j] = self.data[i][j]

        return C   

    # special(dunder) method ---------------------------------
       
    def __eq__(self, other): # 객체끼리 비교할때 서로 다른 객체이면 False를 반환하기에 dunder로 수정한다. 
        if not isinstance(other,Matrix):
            raise TypeError("Other must be a Matrix.")
        if not self.same_shape(other):
            raise ValueError("Other must be a same shape.")
        
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > 1e-10:
                    return False
        return True
    
    def __str__(self): #행렬 출력의 가시성을 높이기 위한 dunder methiod
        result = ""

        for row in self.data:
            result += "[ "
            for value in row:
                result += str(value) + " "
            result += "]\n"
        
        return result
    
    def __add__(self, other): 
        return self.add(other)
    
    def __sub__(self, other):
        return self.subtract(other)
    
    def __matmul__(self, other):
        return self.multiply(other)
    
    def __mul__(self, c):
        return self.scalar_multiply(c)
    
    def __rmul__(self, c):
        return self.scalar_multiply(c)

    
     
    
    def shape(self): 
        return (self.rows,self.cols)
    
    def is_square(self): 
        return self.rows == self.cols
            
    
    def frobenius_norm(self):
        total = 0
        for i in range(self.rows):
            for j in range(self.cols):
                total += self.data[i][j]**2

        return total**0.5
    
    def can_multiply(self,other): 
        
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")

        return self.cols == other.rows
            
    
    def same_shape(self,other): 
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")
        
        return self.shape() == other.shape() 
            
    
        

    
    
    @staticmethod
    def identity(n): 
        if n<=0:
            raise ValueError("Enter an integer value greater than 0.")
        C = [[0]*n for _ in range(n)]

        for i in range(n):
            C[i][i] = 1
        
        return Matrix(C)
    
    @staticmethod
    def zeros(rows,cols): 
        if rows <= 0 or cols <= 0:
            raise ValueError("Enter an integer value greater than 0.")
        return Matrix([[0]*cols for _ in range(rows)])
    
    @staticmethod 
    def diagonal(values):
        if not isinstance(values,list):
            raise TypeError("Matrix data must be a list.")
        if not values:
            raise ValueError("Is empty list.")
        if not all(isinstance(val,(int,float)) for val in values):
            raise TypeError("Matrix elments must be int or float")
        n = len(values)

        C = [[0]*n for _ in range(n)]

        for i in range(n):
            C[i][i] = values[i]

        return Matrix(C)
        
    # Basic Matrix Operations ---------------------------------
    
    def add(self,other): 
        if not self.same_shape(other):
            raise ValueError("Matrices are not equal in size.")
        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = self.data[i][j] + other.data[i][j]

        return Matrix(C)
    
    def subtract(self,other):
        if not self.same_shape(other):
            raise ValueError("Matrices are not equal in size.")
            
        
        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = self.data[i][j] - other.data[i][j]

        return Matrix(C)
    

    def scalar_multiply(self,c): 

        if isinstance(c,bool) or not isinstance(c,(int,float)):
            raise TypeError("Enter only integer or decimal values.")

        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = c * self.data[i][j]

        return Matrix(C)
    

    def transpose(self):
        C = [[0] * self.rows for _ in range(self.cols)]

        for i in range(self.cols):
            for j in range(self.rows):
                C[i][j] = self.data[j][i]
        
        return Matrix(C)
    
    def multiply(self,other): 
        if not self.can_multiply(other):
            raise ValueError("The matrices are not equal in size in" \
            " rows of first matrix and columns second matrix.")
        
        
        cols_other = other.cols
        C = [[0] * cols_other for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(cols_other):
                    for k in range(self.cols):
                        C[i][j] += self.data[i][k] * other.data[k][j]
        
        return Matrix(C)
    
    # Matrix Properties ---------------------------------

    def determinant(self): #가우스 소거법 기반으로 대각성분을 곱해 행렬식을 구한다.
        if not self.is_square(): # 정사각행렬만 행렬식을 구할 수 있다.
            raise ValueError("The determinant must be a square matrix.")
        C = self._copy_data()
        sign = 1
        for k in range(self.rows):
            pivot = C[k][k] 

            #현재 열에서 0이 아닌 pivot 행을 찾는다.
            if abs(pivot) < self.EPS: # float형 부동소수점오차를 방지하기 위한 eps. 
                found = False
                for i in range(k+1,self.rows):
                    #만약 0이 아니라면 행교환을 하여 pivot행을 0이 아닌 값으로 수정한다.
                    if C[i][k] != 0:
                        found = True
                        C[i] , C[k] = C[k], C[i]

                    
                        sign *= -1 # 행교환이 일어나면 부호가 반대가 된다.
                        break
                #0이 아닌 값을 찾지 못했다면 이 행렬의 행렬식은 0이다.
                if not found:
                    return 0
        
            #pivot 아래 성분들을 0으로 만들어 상삼각행렬을 만든다.
            #행렬은 한 행에서 다른행의 배수를 빼도 행렬식에는 변화가 없다.
            for i in range(k+1,self.rows):
                #Ri <- Ri - factor * Rk 를 통해 pivot 아래 성분 C[i][k]를 0으로 만든다.
                #C[i][k]를 0으로 만들기 위한 배수이다.
                factor = C[i][k] / C[k][k]

                for j in range(k,self.cols):
                    C[i][j] -= factor * C[k][j]
        det = sign
        for i in range(self.rows):
            det *= C[i][i]
        
        
        return det
    def inverse(self): # 가우스-조르당 소거법 기반 역행렬 구하기
        if self.determinant() == 0:
            raise ValueError("The inverse matrix does not exist " \
            "because the determinant is zero.")
        
        # left는 A를 I로 바꿔가는 행렬이고,
        # right는 같은 행 연산을 적용해서 A의 역행렬이 되는 행렬이다.
        right = Matrix.identity(self.rows).data
        left = self._copy_data()

        
        for k in range(self.rows):
            pivot = left[k][k]
            
            if pivot == 0:
                found = False
                for i in range(k+1,self.rows):
                    if left[i][k] != 0:
                        found = True
                        left[i] , left[k] = left[k], left[i]
                        right[i], right[k] = right[k], right[i]
                        pivot = left[k][k]
                        break

                #행렬식이 0이면 행렬은 역행렬이 존재하지 않음(singular).
                if not found:
                    raise ValueError("The inverse matrix does not exist" \
                    " because the determinant is zero.")
            #pivot 행을 pivot으로 나누어 pivot값을 1로 만든다.
            for j in range(self.rows):
                left[k][j] = left[k][j] / pivot
                right[k][j] = right[k][j] / pivot
            
            # pivot 열의 다른 행들을 0으로 만든다.
            # left에 한 행 연산을 right에도 똑같이 적용해야 한다.
            # pivot 위아래 행 모두 0이 되게 한다.
            for i in range(self.rows): #행기준
                if i != k:
                    factor = left[i][k]
                    for j in range(self.cols): #열기준
                        left[i][j] -= factor*left[k][j]
                        right[i][j] -= factor*right[k][j]
            
        return Matrix(right)
    
    def row_echelon(self): # REF
        C = self._copy_data()
        
        row = 0

        for col in range(self.cols):
            if self.rows == row:
                break
            
            pivot_row = None
            found = False
            #0이 아닌 pivot행을 찾는다.
            for i in range(row,self.rows):
                if abs(C[i][col]) > self.EPS:
                    found = True
                    pivot_row = i
                    break
            
            #만약 찾지 못했다면 다음 열을 확인한다.
            if not found:
                continue

            C[row], C[pivot_row] = C[pivot_row], C[row]
            
            pivot = C[row][col]

            for i in range(row+1,self.rows):
                factor = C[i][col] / pivot
                for j in range(col,self.cols):
                    C[i][j] -= factor * C[row][j]
            #전 행에서 pivot을 정했던 행은 제외
            row += 1 

        return Matrix(C)
    
    def rref(self):
        C = self._copy_data()

        row = 0

        for col in range(self.cols):

            if row == self.rows:
                break

            pivot_row = None
            found = False

            #0이 아닌 pivot행을 찾는다.
            for i in range(row,self.rows):
                if abs(C[i][col]) > self.EPS:
                    pivot_row = i
                    found = True
                    break

            #만약 찾지 못했다면 다음 열을 확인한다.
            if not found:
                continue
            
            C[pivot_row],C[row] = C[row],C[pivot_row]

            pivot = C[row][col]
            
            #pivot 값을 1로 만들어 준다.
            for j in range(self.cols):
                C[row][j] /= pivot


            for i in range(self.rows):
                if i != row:
                    factor = C[i][col]

                    for j in range(self.cols):
                        C[i][j] -= factor*C[row][j]
            #전 행에서 pivot을 정했던 행은 제외
            row += 1
        
        return Matrix(C)
                
    
    def rank(self): #row echelon을 이용한 rank 계산.
        Row_Echelon = self.row_echelon()
        rank_count = 0

        for i in range(self.rows):
            found = False
            #0이 아닌 행의 수가 rank
            for j in range(self.cols):
                if abs(Row_Echelon.data[i][j]) > self.EPS:
                    found = True
                    break
            if found:
                rank_count += 1
        
        return rank_count
    
    # Linear Systems ---------------------------------

    def augment(self,other):
        if not isinstance(other,Matrix):
            raise TypeError("Other must be a Matrix")
        if self.rows != other.rows:
            raise ValueError("Matrices must be the same number of rows to agument.")
        
        result = []

        for i in range(self.rows):
            result.append(self.data[i] + other.data[i])
        
        return Matrix(result)
    
    def solve_unique(self,b):
        status, rref_augmented = self.analyze_system(b)
        if status != "unique":
            raise ValueError("The system does not have a unique solution.")
        
        C = rref_augmented.data

        
        #마지막 열에서 해 꺼내기
        solution = [[0] for _ in range(self.cols)]

        for i in range(self.rows):
            pivot_col = None
            for j in range(self.cols):
                if abs(C[i][j]) > self.EPS:
                    pivot_col = j
                    break
            
            if pivot_col is not None:
                solution[pivot_col][0] = C[i][-1]
                
        return Matrix(solution)
    
    def analyze_system(self,b):
        if not isinstance(b,Matrix):
            raise TypeError("b must be a Matrix")
        
        if b.cols != 1:
            raise ValueError("b must be a column vector. ")
        
        augmented = self.augment(b)

        rref_augmented = augmented.rref()

        C = rref_augmented.data

        # 모순 행 검사.
        # [0 0 ... 0 | Nonzero] 형태면 0 = Nonzero 이므로 해가 없다.
        for i in range(self.rows):
            all_zero = True
            for j in range(self.cols):
                if abs(C[i][j]) > self.EPS:
                    all_zero = False
                    break
            
            if all_zero and abs(C[i][-1])>self.EPS: 
                return ("no solution",rref_augmented)
        
        #pivot 개수 세기
        #모든 변수 열에 pivot이 있어야 자유변수가 없고 유일해 가능.
        pivot_count = 0

        for i in range(self.rows):
            for j in range(self.cols):
                if abs(C[i][j]) > self.EPS:
                    pivot_count += 1
                    break
        
        if pivot_count == self.cols:
            return ("unique",rref_augmented)
        
        else:
            return ("infinite solutions",rref_augmented)
        
    
    def has_solution(self, b):
        status,_ = self.analyze_system(b)
        return status != "no solution"
    
    # Least Squares -------------------------

    def least_squares(self,b):
        if not isinstance(b, Matrix):
            raise TypeError("b must be a Matrix.")
    
        if b.cols != 1:
            raise ValueError("b must be a column vector.")
    
        if self.rows != b.rows:
            raise ValueError("A and b must have the same number of rows.")

        # normal equation
        # ATAx_hat = ATb
        At = self.transpose()
        normal_A = At @ self
        normal_b = At @ b
        x_hat = normal_A.solve_unique(normal_b)

        return x_hat
    
    def least_squares_prediction(self,b):
        # A의 열의 선형결합중 b에 가장 가까운 백터
        # == 원래 벡터 b를 어떤 공간 안에서 가장 비슷하게 표현한 벡터
        x_hat = self.least_squares(b)

        return self @ x_hat
    
    def residual(self,b):
        # 오차 계산 함수
            
        projection = self.least_squares_prediction(b)
        return b - projection
    

    def residual_norm(self,b):
        return self.residual(b).frobenius_norm()
    
    def is_vector(self):
        return self.rows == 1 or self.cols == 1
    
    #백터를 리스트로 변환한다.
    def _to_vector_list(self):
        if not self.is_vector():
            raise ValueError("Matrix is not a vector.")
        
        values = []

        for i in range(self.rows):
            for j in range(self.cols):
                values.append(self.data[i][j])

        return values
    
    #백터 곱 연산
    def dot(self,other):
        if not isinstance(other,Matrix):
            raise TypeError("other must be a Matrix")
        
        if not self.is_vector() or not other.is_vector():
           raise ValueError("Dot product is only defined for vectors.")
        
        u = self._to_vector_list()
        v = other._to_vector_list()

        if len(u) != len(v):
            raise ValueError("Vectors must have the same size.")
        
        total = 0

        for i in range(len(u)):
            total += u[i] * v[i]

        return total
    

    def normalize(self):
        if not self.is_vector():
            raise ValueError("Normalize is only defined for vectors.")
        
        norm = self.frobenius_norm()

        if abs(norm) < self.EPS:
            raise ValueError("Zero vector cannot be normalized.")
        
        return self.scalar_multiply(1/norm)
    
    #행렬에서 특정 인덱스 열백터만 추출
    def get_column(self,col_index):
        if not isinstance(col_index,int) or isinstance(col_index,bool):
            raise TypeError("Column index must be an integer.")
        
        if col_index < 0 or col_index >= self.cols:
            raise IndexError("Column index out of range.")
        
        column = []

        for i in range(self.rows):
            column.append([self.data[i][col_index]])

        return Matrix(column)
    
    #열백터들을 다시 행렬로 결합.
    @staticmethod
    def from_columns(columns):
        if not isinstance(columns, list):
            raise TypeError("columns must be a list.")

        if not columns:
            raise ValueError("columns cannot be empty.")
        

        for col in columns:
            if not isinstance(col, Matrix):
                raise TypeError("Each column must be a Matrix.")
            #열백터의 열길이는 1.
            if col.cols != 1:
                raise ValueError("Each column must be a column vector.")
            
        row_size = columns[0].rows

        #열백터들은 행길이가 동일해야함.
        for col in columns:
            if col.rows != row_size:
                raise ValueError("All columns must have the same number of rows.")
        
        
        result = []
        #열백터들의 첫번째 요소만 추출한다.
        for i in range(row_size):
            row = []

            for col in columns:
                row.append(col.data[i][0])
            
            result.append(row)
        
        return Matrix(result)
    
    #그램-슈미트 알고리즘
    def gram_schmidt(self):
        if not self.is_full_column_rank():
            raise ValueError("Columns must be linearly independent.")

        column_vectors = []
        
        for col in range(self.cols):
            v = self.get_column(col)

            for q in column_vectors:
                factor = v.dot(q)
                projection = q.scalar_multiply(factor)
                v = v -  projection
            #한 백터가 다른백터로 표현가능하기에 선형종속
            if v.frobenius_norm() < self.EPS:
                raise ValueError("Columns are linearly dependent.")
            
            q = v.normalize()

            column_vectors.append(q)
        
        return Matrix.from_columns(column_vectors)
    
    def is_full_column_rank(self):
        return self.rank() == self.cols
    
    def qr_decomposition(self):
        #열백터들이 선형독립인지 확인.
        if not self.is_full_column_rank():
            raise ValueError("Columns must be linearly independent.")
        # A = QR
        # R = QtA
        Q = self.gram_schmidt()
        R = Q.transpose() @ self
        
        return Q,R
    
    def is_upper_triangular(self):
        
        if not self.is_square():
            raise ValueError("Upper Triangular Matrix must be square. ")
    
        # 상삼각행렬 -> 행 인덱스가 열 인덱스가 큰 구간에서 0
        for i in range(self.rows):
            for j in range(i):
                if abs(self.data[i][j]) > self.EPS:
                    return False
        

        return True
    
    def is_orthonormal(self):

        columns  = []
        for j in range(self.cols):
            v = self.get_column(j)
            if abs(v.frobenius_norm() - 1) > self.EPS:
                return False
            
            columns.append(v)
        # 다른 백터와 곱했을때 0
        for i in range(self.cols):
            v = columns[i]
            for j in range(i+1,self.cols):
                if abs(v.dot(columns[j]))>self.EPS:
                    return False
        
        return True




            
            



    




        




        










            
        



    

    

    



            

        
        
    







        


                

        
        

    


                        


