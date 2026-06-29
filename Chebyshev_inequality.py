import random
import math

# 1. 데이터 생성 1~100 사이의 난수 10,000개 (실험 재료 준비)
data = [random.randint(1,100) for _ in range(10000)]
n = len(data)

# 2. 평군(mu) 계산: 모든 데이터의 합 / 데이터 개수
mean = sum(data) / n

# 3. 표준편차(sigma) 계산: 데이터가 평균에서 얼마나 떨어져 있는가?
# 분산(variance) = (데이터 - 평균)^2의 평균

variance = sum((x - mean) **2 for x in data) / n
std_dev = math.sqrt(variance)

print(f"--- 데이터 분석 결과 ---")
print(f"평균(um) : {mean:.2f}, 표준편차(σ): {std_dev:.2f}")

# 4. 체비쇼프 부등식 검증 (k=5일때라고 가정)
# 이론: P(|X - μ| >= kσ) <= 1/k^2
k = 2
threshold = k * std_dev

# 범위를 벗어나는 데이터(|X - mean| >= k*sigma) 찾기
outloers  = [x for x in data if abs(x - mean) >= threshold]
actual_ratio = len(outloers)/n #벗어나는 데어터의 비율 --> k/n

# 5. 최종 결과 출력 및 이론값 비교
print(f"\n--- 체비쇼프 검증 결과 (k = {k})---")
print(f"이론적 상한선 (1/k^2): {1/k**2:.2%}")
print(f"실제 이탈 데이터 비율: {actual_ratio:.2%}")

if actual_ratio <= 1/(k**2):
    print("\n결과: 체비쇼프 부등식이 성립합니다! ✅")