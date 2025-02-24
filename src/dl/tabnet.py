import os
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch

# ✅ 랜덤 시드 고정
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

# ✅ 데이터 로드 및 전처리
# 여기서는 'ID' 컬럼을 제거하고, '임신 성공 여부'를 레이블로 사용한다고 가정합니다.
train = pd.read_csv('./data/train.csv').drop(columns=['ID'])
test = pd.read_csv('./data/test.csv').drop(columns=['ID'])
sample_submission = pd.read_csv('./data/sample_submission.csv')

label = '임신 성공 여부'
features = [col for col in train.columns if col != label]

# numpy 배열로 변환
X_train = train[features].values.astype(np.float32)
y_train = train[label].values.astype(np.int64)  # TabNetClassifier는 정수형 label 사용
X_test = test[features].values.astype(np.float32)

# ✅ 5-Fold 설정
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 예측 결과 저장용 배열
oof_preds = np.zeros(len(train))
test_preds = np.zeros((len(test), n_splits))

# ✅ KFold Cross-Validation으로 TabNet 모델 학습
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[valid_idx], y_train[valid_idx]
    
    clf = TabNetClassifier(
        seed=42,
        verbose=1,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 모델 학습
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Validation 예측 및 AUC 평가
    val_preds = clf.predict_proba(X_val)[:, 1]
    oof_preds[valid_idx] = val_preds
    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
    
    # Test 데이터 예측
    test_fold_preds = clf.predict_proba(X_test)[:, 1]
    test_preds[:, fold] = test_fold_preds

# 전체 OOF AUC 평가
overall_auc = roc_auc_score(y_train, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

# Fold별 테스트 예측 평균
final_test_preds = test_preds.mean(axis=1)

# ✅ 예측 결과 파일 저장
sub_dir = './submissions'
os.makedirs(sub_dir, exist_ok=True)

submission = sample_submission.copy()
submission['probability'] = final_test_preds
submission_file = os.path.join(sub_dir, 'tabnet_predictions.csv')
submission.to_csv(submission_file, index=False)
print(f"\n✅ TabNet 예측 결과가 {submission_file}에 저장되었습니다.")
