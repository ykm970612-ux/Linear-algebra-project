class Matrix:
    def __init__(self,A): # 행렬 기본정보 초기화.
        self._validate_matrix(A)


        self.data = A
        self.rows = len(A)
        self.cols = len(A[0])

    def _validate_matrix(self,A): # 행렬의 조건 확인.
        if not A: # 빈 행렬인지 확인 
            raise ValueError("Empty Matrix")
        if not isinstance(A,list): # 구조가 리스트인지 확인.
            raise TypeError("Matrix data must be a list")

        if not all(isinstance(row,list) for row in A): #행의 구조가 리스트 인지 확인.
            raise TypeError("Each row must be a list")

        col_size = len(A[0])

        if col_size == 0: #열의 크기가 0인지 확인.
            raise ValueError("Matrix row cannot be empty")

        if not all(len(row) == col_size for row in A): #모든 행이 같은 크기인지 확인.
            raise ValueError("All row must have the same length")
        
        if not all(isinstance(value,(int,float)) and not isinstance(value,bool)
                   for row in A
                   for value in row
        ): # 모든 요소가 숫자인지 확인.
            raise TypeError("Matrix elments must be numbers")
        
    
    def _copy_data(self): #배열의 요소를 복사해 새로운 객체생성.

        rows = len(self.data)
        cols = len(self.data[0])
        C = [[0]*cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                C[i][j] = self.data[i][j]

        return C      
    def __eq__(self, other):
        if not isinstance(other,Matrix):
            raise TypeError("Other must be a Matrix.")
        if not self.same_shape(other):
            raise ValueError("Other must be a same shape.")
        
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > 1e-10:
                    return False
        return True
    
    # special(dunder) method
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

     
    
    def shape(self): #행렬의 크기를 반환.
        return (self.rows,self.cols)
    
    def is_square(self): # 정사각행렬 여부확인.
        if self.rows == self.cols:
            return True
        
        return False
    
    def can_multiply(self,other): # 행렬 곱이 가능한지 확인.
        
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")

        if self.cols == other.rows:
            return True
    
        return False
    def same_shape(self,other): # 같은 크기의 행렬인지 확인.
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")
        if self.shape() == other.shape()  :
            return True
    
        return False

    def __str__(self): #객체출력함수
        result = ""

        for row in self.data:
            result += "[ "
            for value in row:
                result += str(value) + " "
            result += "]\n"
        
        return result
    
    @staticmethod
    def identity(n): #nxn 크기의 항등행렬 생성.
        if n<=0:
            raise ValueError("Enter an integer value greater than 0.")
        C = [[0]*n for _ in range(n)]

        for i in range(n):
            C[i][i] = 1
        
        return Matrix(C)
    
    @staticmethod
    def zeros(rows,cols): # rows x cols 크기의 영행렬 생성. 
        if rows <= 0 or cols <= 0:
            raise ValueError("Enter an integer value greater than 0.")
        return Matrix([[0]*cols for _ in range(rows)])
    
    @staticmethod # 대각행렬 생성.
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
        

    
    def add(self,other): # 행렬의 합
        if not self.same_shape(other):
            raise ValueError("Matrices are not equal in size.")
        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = self.data[i][j] + other.data[i][j]

        return Matrix(C)
    
    def subtract(self,other): # 행렬의 차
        if not self.same_shape(other):
            raise ValueError("Matrices are not equal in size.")
            
        
        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = self.data[i][j] - other.data[i][j]

        return Matrix(C)
    

    def scalar_multiply(self,c): # 스칼라 행렬 곱

        if isinstance(c,bool) or not isinstance(c,(int,float)):
            raise TypeError("Enter only integer or decimal values.")

        
        C = [[0] * self.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                C[i][j] = c * self.data[i][j]

        return Matrix(C)
    

    def transpose(self): # 행렬 전치
        C = [[0] * self.rows for _ in range(self.cols)]

        for i in range(self.cols):
            for j in range(self.rows):
                C[i][j] = self.data[j][i]
        
        return Matrix(C)
    
    def multiply(self,other): # 행렬 곱
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
    

    def determinant(self): #가우스 소거법 기반 행렬식
        if not self.is_square():
            raise ValueError("The determinant must be a square matrix.")
        C = self._copy_data()
        sign = 1
        eps = 1e-10
        for k in range(self.rows):
            pivot = C[k][k]

            if abs(pivot) < eps:
                found = False
                for i in range(k+1,self.rows):
                    if C[i][k] != 0:
                        found = True
                        C[i] , C[k] = C[k], C[i]

                    
                        sign *= -1
                        break
                if not found:
                    return 0
        

            for i in range(k+1,self.rows):
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

                if not found:
                    raise ValueError("The inverse matrix does not exist" \
                    " because the determinant is zero.")
                
            for j in range(self.rows):
                left[k][j] = left[k][j] / pivot
                right[k][j] = right[k][j] / pivot
            

            for i in range(self.rows): #행기준
                if i != k:
                    factor = left[i][k]
                    for j in range(self.cols): #열기준
                        left[i][j] -= factor*left[k][j]
                        right[i][j] -= factor*right[k][j]
            
        return Matrix(right)
    
    def row_echelon(self): # REF
        C = self._copy_data()
        
        eps =  1e-10 # 나눗셈이 들어가면 float 오차가 생길 수 있음
        row = 0

        for col in range(self.cols):
            if self.rows == row:
                break

            pivot_row = None
            found = False
            for i in range(row,self.rows):
                if abs(C[i][col]) > eps:
                    found = True
                    pivot_row = i
                    break
            
            if not found:
                continue

            C[row], C[pivot_row] = C[pivot_row], C[row]
            
            pivot = C[row][col]

            for i in range(row+1,self.rows):
                factor = C[i][col] / pivot
                for j in range(col,self.cols):
                    C[i][j] -= factor * C[row][j]

            row += 1 

        return Matrix(C)
    
    def rank(self): #Rank 수 반환
        Row_Echelon = self.row_echelon()
        rank_count = 0

        for i in range(self.rows):
            found = False
            for j in range(self.cols):
                if Row_Echelon.data[i][j] != 0:
                    found = True
                    break
            if found:
                rank_count += 1
        
        return rank_count

        
        
    
    
    

    

    




    
        
    

    
    
    
        

        



    


        
        

       
        


































    







        


                

        
        

    


                        


