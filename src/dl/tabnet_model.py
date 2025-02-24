import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv('C:/Users/okpl8/Documents/project_LG/data/train.csv') 
test = pd.read_csv('C:/Users/okpl8/Documents/project_LG/data/test.csv')
submit=pd.read_csv('C:/Users/okpl8/Documents/project_LG/data/sample_submission.csv')

train = train.drop('ID', axis = 1)
test = test.drop('ID', axis = 1)

train.drop(columns=['PGD 시술 여부','PGS 시술 여부','난자 해동 경과일','배아 해동 경과일',
            '임신 시도 또는 마지막 임신 경과 연수','착상 전 유전 검사 사용 여부'],inplace=True)
test.drop(columns=['PGD 시술 여부','PGS 시술 여부','난자 해동 경과일','배아 해동 경과일',
            '임신 시도 또는 마지막 임신 경과 연수','착상 전 유전 검사 사용 여부'],inplace=True)

train["특정 시술 유형"] = train["특정 시술 유형"].fillna("Unknown").astype(str)
test["특정 시술 유형"] = test["특정 시술 유형"].fillna("Unknown").astype(str)

def categorize_treatment(x):
    first_part = x.split(":")[0]  # 첫 번째 값 가져오기

    if first_part == "ICSI":
        return "ICSI"
    elif first_part == "IVF":
        return "IVF"
    elif first_part in ["IUI", "ICI"]:
        return "IUI"
    else:
        return "Unknown"

# ✅ 그룹화 적용 (train & test)
train["특정 시술 유형"] = train["특정 시술 유형"].apply(categorize_treatment)
test["특정 시술 유형"] = test["특정 시술 유형"].apply(categorize_treatment)

mode_cols = ["착상 전 유전 진단 사용 여부", "배아 생성 주요 이유", "해동 난자 수", "저장된 신선 난자 수", "기증자 정자와 혼합된 난자 수",
             "기증 배아 사용 여부", "대리모 여부"]

for col in mode_cols:
    if col in train.columns and col in test.columns:  # train과 test 모두 해당 컬럼이 있을 때만 실행
        mean_value = train[col].mode()[0]  # 🚨 Train 데이터의 최빈값
        train[col] = train[col].fillna(mean_value)  # inplace=False 방식으로 수정
        test[col] = test[col].fillna(mean_value)    # inplace=False 방식으로 수정
    else:
        print(f"컬럼 {col}이(가) train 또는 test 데이터에 없습니다.")
        
mode_cols = ["단일 배아 이식 여부", "미세주입 후 저장된 배아 수", "해동된 배아 수", "동결 배아 사용 여부", "신선 배아 사용 여부",
             "난자 채취 경과일", "난자 혼합 경과일","저장된 배아 수","미세주입에서 생성된 배아 수","미세주입 배아 이식 수", "미세주입된 난자 수"]

for col in mode_cols:
    if col in train.columns and col in test.columns:  # train과 test 모두 해당 컬럼이 있을 때만 실행
        mean_value = train[col].mode()[0]  # 🚨 Train 데이터의 최빈값
        train[col] = train[col].fillna(mean_value)  # inplace=False 방식으로 수정
        test[col] = test[col].fillna(mean_value)    # inplace=False 방식으로 수정
    else:
        print(f"컬럼 {col}이(가) train 또는 test 데이터에 없습니다.")
        
median_cols = ['총 생성 배아 수','','이식된 배아 수','수집된 신선 난자 수','혼합된 난자 수','파트너 정자와 혼합된 난자 수','배아 이식 경과일']
for col in median_cols:
    if col in train.columns and col in test.columns:  # train과 test 모두 해당 컬럼이 있을 때만 실행
        mean_value = train[col].median()  # 🚨 Train 데이터의 중앙값
        train[col] = train[col].fillna(mean_value)  # inplace=False 방식으로 수정
        test[col] = test[col].fillna(mean_value)    # inplace=False 방식으로 수정
    else:
        print(f"컬럼 {col}이(가) train 또는 test 데이터에 없습니다.")
        
train.drop(columns=['배란 유도 유형','난자 기증자 나이','정자 기증자 나이'],inplace=True)
test.drop(columns=['배란 유도 유형','난자 기증자 나이','정자 기증자 나이'],inplace=True)

train.drop(columns=['난자 채취 경과일','불임 원인 - 여성 요인','불임 원인 - 정자 면역학적 요인'],inplace=True)
test.drop(columns=['난자 채취 경과일','불임 원인 - 여성 요인','불임 원인 - 정자 면역학적 요인'],inplace=True)

# 배아 생성 주요 이유 그룹화 함수
def categorize_embryo_reason(x):
    if "현재 시술용" in x:
        return "Current"
    elif "배아 저장용" in x:
        return "Storage"
    elif "기증용" in x:
        return "Donation"
    elif "난자 저장용" in x:
        return "Egg Storage"
    elif "연구용" in x:
        return "Research"
    else:
        return "Unknown"

train["배아 생성 주요 이유"] = train["배아 생성 주요 이유"].apply(categorize_embryo_reason)
test["배아 생성 주요 이유"] = test["배아 생성 주요 이유"].apply(categorize_embryo_reason)

from sklearn.preprocessing import LabelEncoder

label_col=['시술 시기 코드','시술 당시 나이','총 시술 횟수','클리닉 내 총 시술 횟수','IVF 시술 횟수','DI 시술 횟수',
           '총 임신 횟수','IVF 임신 횟수','DI 임신 횟수','총 출산 횟수','IVF 출산 횟수','DI 출산 횟수']
# Label Encoding

for col in label_col:
    # Label Encoder 생성
    le = LabelEncoder()
    
    # Train 데이터에 Label Encoding 적용
    train[col] = le.fit_transform(train[col].astype(str))
    
    # Test 데이터에 동일한 Label Encoding 적용
    test[col] = le.transform(test[col].astype(str))  # Test에는 transform만 적용

# 인코딩 확인
print(train[label_col].head())
print(test[label_col].head())

from sklearn.preprocessing import OneHotEncoder

onehot_col=['시술 유형','특정 시술 유형','배아 생성 주요 이유','난자 출처','정자 출처'] 

# OneHotEncoder 생성
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'는 더미 변수 함정 방지

# Train 데이터에 One-Hot Encoding 적용
train_encoded = ohe.fit_transform(train[onehot_col])

# Test 데이터에 동일한 One-Hot Encoding 적용
test_encoded = ohe.transform(test[onehot_col])

# 인코딩된 데이터를 DataFrame으로 변환하고 기존 데이터와 합치기
train_encoded_df = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out(onehot_col))
test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(onehot_col))

train = pd.concat([train.drop(columns=onehot_col), train_encoded_df], axis=1)
test = pd.concat([test.drop(columns=onehot_col), test_encoded_df], axis=1)

train.drop(columns=['남성 주 불임 원인','IVF 시술 횟수','IVF 임신 횟수','IVF 출산 횟수','미세주입에서 생성된 배아 수',
                    '파트너 정자와 혼합된 난자 수'],inplace=True)
test.drop(columns=['남성 주 불임 원인','IVF 시술 횟수','IVF 임신 횟수','IVF 출산 횟수','미세주입에서 생성된 배아 수',
                    '파트너 정자와 혼합된 난자 수'],inplace=True)

train.drop(columns=['여성 주 불임 원인','DI 임신 횟수','DI 출산 횟수','혼합된 난자 수'],inplace=True)
test.drop(columns=['여성 주 불임 원인','DI 임신 횟수','DI 출산 횟수','혼합된 난자 수'],inplace=True)

train.drop(columns=['부부 부 불임 원인','총 시술 횟수','총 출산 횟수','수집된 신선 난자 수','동결 배아 사용 여부',
                    '신선 배아 사용 여부'],inplace=True)
test.drop(columns=['부부 부 불임 원인','총 시술 횟수','총 출산 횟수','수집된 신선 난자 수','동결 배아 사용 여부',
                    '신선 배아 사용 여부'],inplace=True)

# 상관행렬 계산
corr_matrix = train.corr()

# 상관관계가 0.7 이상인 값만 선택 (자기 자신과의 상관관계 제외)
high_corr = corr_matrix[(corr_matrix > 0.7) & (corr_matrix < 1)]

# 결과를 변수와 해당 변수와 0.7 이상인 상관관계를 가진 다른 변수들로 출력
for col in high_corr.columns:
    related_vars = high_corr[col].dropna()  # 결측치 제외
    related_vars = related_vars[related_vars >= 0.7]  # 0.7 이상인 값만 선택
    if len(related_vars) > 0:  # 상관관계가 0.7 이상인 값이 있을 경우
        print(f"{col}: {', '.join([f'{var}: {related_vars[var]:.2f}' for var in related_vars.index])}")
        
mode_cols = ["단일 배아 이식 여부", "미세주입 후 저장된 배아 수", "해동된 배아 수", "동결 배아 사용 여부", "신선 배아 사용 여부",
             "난자 채취 경과일", "난자 혼합 경과일","저장된 배아 수","미세주입에서 생성된 배아 수","미세주입 배아 이식 수", "미세주입된 난자 수"]

for col in mode_cols:
    if col in train.columns and col in test.columns:  # train과 test 모두 해당 컬럼이 있을 때만 실행
        mean_value = train[col].mode()[0]  # 🚨 Train 데이터의 최빈값
        train[col] = train[col].fillna(mean_value)  # inplace=False 방식으로 수정
        test[col] = test[col].fillna(mean_value)    # inplace=False 방식으로 수정
    else:
        print(f"컬럼 {col}이(가) train 또는 test 데이터에 없습니다.")
        
mode_cols = ["착상 전 유전 진단 사용 여부", "배아 생성 주요 이유", "해동 난자 수", "저장된 신선 난자 수", "기증자 정자와 혼합된 난자 수",
             "기증 배아 사용 여부", "대리모 여부"]

for col in mode_cols:
    if col in train.columns and col in test.columns:  # train과 test 모두 해당 컬럼이 있을 때만 실행
        mean_value = train[col].mode()[0]  # 🚨 Train 데이터의 최빈값
        train[col] = train[col].fillna(mean_value)  # inplace=False 방식으로 수정
        test[col] = test[col].fillna(mean_value)    # inplace=False 방식으로 수정
    else:
        print(f"컬럼 {col}이(가) train 또는 test 데이터에 없습니다.")
        
drop_features = [
    "배아 생성 주요 이유_Donation", "배아 생성 주요 이유_Egg Storage", "불임 원인 - 자궁경부 문제",
    "불임 원인 - 정자 운동성", "특정 시술 유형_IUI", "난자 혼합 경과일", "해동 난자 수",
    "대리모 여부", "불임 원인 - 정자 형태"
]
train.drop(columns=drop_features, inplace=True)
test.drop(columns=drop_features, inplace=True)

from sklearn.model_selection import train_test_split

# ✅ 데이터 분할
X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve


###########################################
# 1. Focal Loss 구현
###########################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 확률 변환
        inputs = inputs[:, 1]  # 🔥 클래스 1의 확률만 사용
        targets = targets.float().view(-1)  # 🔥 1D 벡터로 변환

        inputs = inputs.clamp(min=1e-7, max=1-1e-7)  # log(0) 방지
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # 정답 확률 계산
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        return loss.mean() if self.reduction == 'mean' else loss.sum()
focal_loss = FocalLoss(alpha=0.5, gamma=1.0, reduction='mean')

###########################################
# 2. 데이터 준비
###########################################
# X_train, y_train, X_test, test는 이미 전처리된 DataFrame이라고 가정
X_train_tab = X_train.values.astype(np.float32)
y_train_tab = y_train.values.astype(np.float32).squeeze()  # 1D 변환
X_test_tab = X_test.values.astype(np.float32)
X_submission_tab = test.values.astype(np.float32)  # 제출용 test 데이터

###########################################
# 3. 5-Fold Stratified CV를 이용한 TabNet 학습 (GPU 사용)
###########################################
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stacking_train_meta_tab = np.zeros((X_train_tab.shape[0], 1))
stacking_test_meta_tab = np.zeros((X_test_tab.shape[0], 1))
stacking_submission_meta_tab = np.zeros((X_submission_tab.shape[0], 1))  # 제출 예측용

fold_idx = 0
for train_index, val_index in skf.split(X_train_tab, y_train_tab):
    print(f"\n=== TabNet Fold {fold_idx+1} ===")
    X_train_fold_tab, X_val_fold_tab = X_train_tab[train_index], X_train_tab[val_index]
    y_train_fold_tab, y_val_fold_tab = y_train_tab[train_index], y_train_tab[val_index]

    # TabNetClassifier 생성 (GPU 사용, focal loss 적용)
    tabnet_clf = TabNetClassifier(
        device_name='cuda',  # GPU 사용 명시
        optimizer_params=dict(lr=0.001),
        scheduler_params={"step_size": 6, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42,
        verbose=1
    )

    # TabNet 학습 (loss_fn에 FocalLoss 직접 지정)
    tabnet_clf.fit(
        X_train_fold_tab, y_train_fold_tab,
        eval_set=[(X_val_fold_tab, y_val_fold_tab)],
        eval_metric=['auc'],
        max_epochs=50,
        patience=10,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        loss_fn=focal_loss  # 🔥 Focal Loss 적용
    )

    # 검증 Fold에 대한 예측 (메타 데이터로 사용)
    stacking_train_meta_tab[val_index, 0] = tabnet_clf.predict_proba(X_val_fold_tab)[:, 1]

    # X_test에 대한 예측 (각 Fold 예측을 평균)
    stacking_test_meta_tab[:, 0] += tabnet_clf.predict_proba(X_test_tab)[:, 1] / skf.n_splits

    # 제출용 test 데이터에 대한 예측 (각 Fold 예측을 평균)
    stacking_submission_meta_tab[:, 0] += tabnet_clf.predict_proba(X_submission_tab)[:, 1] / skf.n_splits

    fold_idx += 1

###########################################
# 4. TabNet 성능 평가
###########################################
tabnet_train_roc_auc = roc_auc_score(y_train, stacking_train_meta_tab.ravel())
print(f"\nTabNet Training ROC-AUC: {tabnet_train_roc_auc:.4f}")

tabnet_test_roc_auc = roc_auc_score(y_test, stacking_test_meta_tab.ravel())
print(f"TabNet Test ROC-AUC: {tabnet_test_roc_auc:.4f}")

precision_tab, recall_tab, thresholds_tab = precision_recall_curve(y_test, stacking_test_meta_tab.ravel())
f1_scores_tab = (2 * precision_tab * recall_tab) / (precision_tab + recall_tab + 1e-8)
optimal_threshold_tab = thresholds_tab[np.argmax(f1_scores_tab)]
print(f"TabNet Optimal Threshold: {optimal_threshold_tab:.4f}")

Y_test_pred_tab = (stacking_test_meta_tab.ravel() > optimal_threshold_tab).astype(int)
print("\n=== TabNet Classification Report ===")
print(classification_report(y_test, Y_test_pred_tab))

###########################################
# 5. Kaggle 제출 파일 생성 (TabNet)
###########################################
submission_filename = f'./data/submission_tabnet_focal_{tabnet_test_roc_auc:.4f}.csv'
sample_submission = pd.read_csv('./data/sample_submission.csv')
sample_submission['probability'] = stacking_submission_meta_tab.ravel()
sample_submission.to_csv(submission_filename, index=False)
print(f"\n✅ Submission file saved: {submission_filename}")
