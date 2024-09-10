import numpy as np
import pandas as pd
import statsmodels.api as sm

data=  pd.read_table('C:/Users/USER/Documents/LS빅데이터스쿨/LSbigdata-project1/data/leukemia_remission.txt', delimiter='\t')
print(data.head())  # 데이터 확인

#종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미
# 독립변수:
# 골수의 세포성 (CELL)
# 골수편의 백혈구 비율 (SMEAR)
# 골수의 백혈병 세포 침투 비율 (INFIL)
# 골수 백혈병 세포의 라벨링 인덱스 (LI)
# 말초혈액의 백혈병 세포 수 (BLAST)
# 치료 시작 전 최고 체온 (TEMP)


# 문제1. 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
X = data[['CELL', 'SMEAR', 'INFIL', 'LI', 'BLAST', 'TEMP']]  # Independent variables
y = data['REMISS'] 

X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
result = logit_model.fit()

logit_summary = result.summary()
logit_summary

# 문제2. 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.

llr_pvalue = result.llr_pvalue #우도비 검정의 p-value : 0.04670
ll_null = result.llnull  #  독립 변수 없이 상수항(intercept)만을 포함한 귀무모형의 로그 우도반환: -17.1858
ll_model = result.llf  # 독립 변수를 포함한 모델로, 독립 변수들이 종속 변수에 영향을 미친다고 가정: (-10.79692
lr_stat = 2 * (ll_model - ll_null) # LR statistic: 12.7779  클수록 독립 변수를 추가한 모델이 종속 변수를 더 잘 설명한다

llr_pvalue, lr_stat #  p-value가 0.05보다 작으므로, 95% 신뢰수준에서 귀무가설을 기각
from scipy.stats import chi2
# −2(ℓ(𝛽)̂ (0) − ℓ(𝛽)̂ )  =  -2*(-17.186+10.797)  = 12.779
1 - chi2.cdf(12.779, df=6)  # 0.0467



# 문제3. 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
p_values = result.pvalues
p_values[p_values < 0.2]
# 2개.  LI ,  TEMP

# 문제4. 다음 환자에 대한 오즈는 얼마인가요?


# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9
coefficients = result.params
patient_data = {
    'const': 1,  # Intercept
    'CELL': 0.65,  # 65%
    'SMEAR': 0.45,  # 45%
    'INFIL': 0.55,  # 55%
    'LI': 1.2,  # LI value
    'BLAST': 1.1,  # BLAST value in cells/μL
    'TEMP': 0.9  # TEMP value (already scaled)
}

log_odds = sum(coefficients[var] * patient_data[var] for var in patient_data)

odds = np.exp(log_odds)

log_odds, odds # -3.2656, 0.03817

# 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
prob = 1 / (1 + np.exp(-log_odds))
prob # 0.036767

# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.

temp_coef = coefficients['TEMP'] # -100.17340

temp_odds_ratio = np.exp(temp_coef)
temp_odds_ratio  # 3.13e-44 - > 0에 가까운 값입니다. 이는 체온이 1단위 상승할 때 백혈병 세포가 관측되지 않을 확률이 (오즈비만큼 변동)거의 없어지는 것을 의미 ->  온도가 높아질수록 백혈병 세포가 관측될 확률 높아짐.

# 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
cell_coef = coefficients['CELL']
cell_se = result.bse['CELL']  # bse는 **표준 오차(Standard Errors, SE)가 저장된 속성

z_value = 2.576 # 99% 신뢰구간에서 z값

lower_log_odds = cell_coef - z_value * cell_se
upper_log_odds = cell_coef + z_value * cell_se

lower_odds_ratio = np.exp(lower_log_odds)
upper_odds_ratio = np.exp(upper_log_odds)

lower_odds_ratio, upper_odds_ratio # 0에 가까운수, 엄청 큰 수.

# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix
predicted_probabilities = result.predict(X)
predicted_classes = (predicted_probabilities >= 0.5).astype(int)
conf_matrix = confusion_matrix(y, predicted_classes)
conf_matrix  

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 혼동행렬 데이터
conf_matrix = [[15, 3], [4, 5]]

# 혼동행렬 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 문제 9. 해당 모델의 Accuracy는 얼마인가요?


accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
accuracy # 정확도: 약 74.07%

# 문제 10. 해당 모델의 F1 Score를 구하세요.
from sklearn.metrics import f1_score

f1 = f1_score(y, predicted_classes)
f1  # 0.58823