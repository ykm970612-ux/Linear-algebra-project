class Matrix:
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

    # special(dunder) method
       
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
        if self.rows == self.cols:
            return True
        
        return False
    
    def can_multiply(self,other): 
        
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")

        if self.cols == other.rows:
            return True
    
        return False
    def same_shape(self,other): 
        if not isinstance(other, Matrix):
            raise TypeError("other must be a Matrix")
        if self.shape() == other.shape()  :
            return True
    
        return False

    
    
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
    

    def determinant(self): #가우스 소거법 기반으로 대각성분을 곱해 행렬식을 구한다.
        if not self.is_square(): # 정사각행렬만 행렬식을 구할 수 있다.
            raise ValueError("The determinant must be a square matrix.")
        C = self._copy_data()
        sign = 1
        eps = 1e-10 #부동소수점 오차 방지를 위한 eps
        for k in range(self.rows):
            pivot = C[k][k] 

            #현재 열에서 0이 아닌 pivot 행을 찾는다.
            if abs(pivot) < eps: # float형 부동소수점오차를 방지하기 위한 eps. 
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
            #pivot 행을 pivot으로 나누어 pivot값을 0으로 만든다.
            for j in range(self.rows):
                left[k][j] = left[k][j] / pivot
                right[k][j] = right[k][j] / pivot
            
            # pivot 열의 다른 행들을 0으로 만든다.
            # left에 한 행 연산을 right에도 똑같이 적용해야 한다.
            for i in range(self.rows): #행기준
                if i != k:
                    factor = left[i][k]
                    for j in range(self.cols): #열기준
                        left[i][j] -= factor*left[k][j]
                        right[i][j] -= factor*right[k][j]
            
        return Matrix(right)
    
    def row_echelon(self): # REF
        C = self._copy_data()
        
        eps =  1e-10 #float형의 부동소수점 오차를 방지하기 위함.
        row = 0

        for col in range(self.cols):
            if self.rows == row:
                break
            
            pivot_row = None
            found = False
            #0이 아닌 pivot행을 찾는다.
            for i in range(row,self.rows):
                if abs(C[i][col]) > eps:
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
    
    def rank(self): #row echelon을 이용한 rank 계산.
        Row_Echelon = self.row_echelon()
        rank_count = 0

        for i in range(self.rows):
            found = False
            #0이 아닌 행의 수가 rank
            for j in range(self.cols):
                if Row_Echelon.data[i][j] != 0:
                    found = True
                    break
            if found:
                rank_count += 1
        
        return rank_count

        
        
    
    
    

    

    




    
        
    

    
    
    
        

        



    


        
        

       
        


































    







        


                

        
        

    


                        


