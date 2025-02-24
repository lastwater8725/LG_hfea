import os
import pandas as pd
import random
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import KFold
from sklearn.calibration import IsotonicRegression
import torch

# ✅ FocalLoss를 별도의 파일에서 import
from src.focal_loss import FocalLoss

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# ✅ 데이터 로드
train = pd.read_csv('./data/train.csv').drop(columns=['ID'])
test = pd.read_csv('./data/test.csv').drop(columns=['ID'])
sample_submission = pd.read_csv('./data/sample_submission.csv')

label = '임신 성공 여부'

# 5-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 결과 저장용 리스트
oof_preds = np.zeros(len(train))
test_preds = np.zeros((len(test), 5))

# ✅ AutoGluon 하이퍼파라미터 설정 (lambda 제거)
hyperparams = {
    'GBM': {'num_boost_round': 500, 'objective': 'binary'},  # LightGBM
    'CAT': {'iterations': 2000, 'depth': 6, 'task_type': 'GPU'},  # CatBoost
    'XGB': {
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'scale_pos_weight': 2  # 불균형 데이터 가중치 조절
    },
    'TABNET': {
        'epochs': 20,  # TabNet 학습 반복 횟수
        'custom_loss_function': FocalLoss(),  # ✅ TabNet에 Focal Loss 적용
        'optimizer': 'adam',  # 옵티마이저
        'virtual_batch_size': 128,  # 배치 크기
        'momentum': 0.02,
        'gamma': 1.3,
    }
}


# ✅ 5-Fold Cross Validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"Training fold {fold + 1}/5...")

    train_fold = train.iloc[train_idx]
    valid_fold = train.iloc[valid_idx]

    predictor = TabularPredictor(
        label='임신 성공 여부',
        problem_type='binary',
        eval_metric='roc_auc'
    ).fit(
        train_data=TabularDataset(train_fold),
        presets='best_quality',
        time_limit=3600 * 10,
        num_stack_levels=5,
        num_gpus=1,
        use_bag_holdout=True,  # ✅ Bagging 모드 해결
        hyperparameters=hyperparams
    )

    # Out-of-Fold 예측 저장
    oof_preds[valid_idx] = predictor.predict_proba(TabularDataset(valid_fold))[:, 1]

    # Test 데이터 예측 저장
    test_preds[:, fold] = predictor.predict_proba(TabularDataset(test))[:, 1]

# ✅ Test Prediction: 5개 Fold 평균
final_test_preds = test_preds.mean(axis=1)

# ✅ Isotonic Calibration 적용
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(oof_preds, train[label])
calibrated_preds = iso_reg.transform(final_test_preds)

# ✅ 결과 저장 및 제출
sample_submission['probability'] = calibrated_preds
sample_submission.to_csv('./submissions/autogluon_focal_loss_calibrated.csv', index=False)

print("✅ AutoGluon 학습 완료! Focal Loss + 5Fold CV + Isotonic Calibration 적용됨.")
