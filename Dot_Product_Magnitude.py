from math import sqrt

def dot_product(v1,v2):
    if len(v1) != len(v2):
        print("에러: 두 벡터의 길이가 다릅니다!")
        return None
    value = 0
    for i in range(len(v1)):
        value += (v1[i] * v2[i])
    return value

def magnitude(v):
    return sqrt(sum(x**2 for x in v))

def cosine_similarity(v1, v2):
    return dot_product(v1,v2) / (magnitude(v1)  * magnitude(v2))

while True:
    print(" ---백터의 유사도 ---")

    try:

        v1 = [float(x) for x in input("첫 번째 벡터를 입력하세요 (예: 1 2 3): ").split()]

        v2 = [float(x) for x in input("두 번째 벡터를 입력하세요 (예: 1 2 3): ").split()]

        if len(v1) != len(v2) :
            print("!!에러 : 두 백터의 차원이 다릅니다. ")
            continue

        simmilarity = cosine_similarity(v1,v2)
        print(f"두 백터의 유사도 --> {simmilarity}")

        if simmilarity >0.9:
            print("두 백터는 매우 비슷한 방향을 향하고 있음")
        elif simmilarity > 0:
            print("두 백터는 어느정도 상관관계가 있습니다. ")
        else:
            print("두 백터는 관계가 없거나 반대입니다 ")
        

    except ZeroDivisionError:
        print("❌ 에러: 크기가 0인 벡터는 계산할 수 없습니다.")
    
    except ValueError:
        print("❌ 에러: 숫자만 입력해 주세요.")
    
    
    if input("\n계속하시겠습니까? (y/n): ").lower() == 'n':
        print("프로그램을 종료합니다.")
        break