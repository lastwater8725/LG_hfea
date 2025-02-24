import os
import pandas as pd
import random
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import KFold
from sklearn.calibration import IsotonicRegression
import torch

import warnings
warnings.filterwarnings("ignore")

# ✅ 랜덤 시드 고정
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

# ✅ 5-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ✅ 결과 저장용 리스트
oof_preds = np.zeros(len(train))
test_preds = np.zeros((len(test), 5))

# ✅ AutoGluon 하이퍼파라미터 설정 (FT_TRANSFORMER 사용)
hyperparams = {
    'GBM': {'num_boost_round': 500, 'objective': 'binary'},  
    #'CAT': {'iterations': 2000, 'depth': 6, 'task_type': 'GPU'},  
    'XGB': {
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'scale_pos_weight': 2
    },
    'FT_TRANSFORMER': {
        "model.ft_transformer.embedding_arch": ["linear"],  
        "env.batch_size": 128,
        "env.per_gpu_batch_size": 128,
        "env.num_workers": 0,
        "optimization.max_epochs": 50,  # 조기 종료 고려
        "optimization.weight_decay": 1.0e-5,
        "optimization.lr_schedule": "polynomial_decay",
        "optimization.patience": 10,  # 10 에포크 동안 개선 없으면 중단
        "optimization.top_k": 3,
        "env.num_gpus": 1
    },
    #TABPFN': {}
}

# ✅ 5-Fold Cross Validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"Training fold {fold + 1}/5...")

    train_fold = train.iloc[train_idx]
    valid_fold = train.iloc[valid_idx]

    predictor = TabularPredictor(
        label=label,
        problem_type='binary',
        eval_metric='roc_auc'
    ).fit(
        train_data=TabularDataset(train_fold),
        presets='high_quality',
        time_limit=3600 * 5,
        num_stack_levels=2,
        num_gpus=1,
        use_bag_holdout=True,
        hyperparameters=hyperparams
    )

    # ✅ Out-of-Fold 예측 저장
    pred_probs = predictor.predict_proba(TabularDataset(valid_fold))
    if isinstance(pred_probs, pd.DataFrame):
        oof_preds[valid_idx] = pred_probs.iloc[:, 1]  
    else:
        oof_preds[valid_idx] = pred_probs[:, 1]

    # ✅ Test 데이터 예측 저장
    pred_probs_test = predictor.predict_proba(TabularDataset(test))
    if isinstance(pred_probs_test, pd.DataFrame):
        test_preds[:, fold] = pred_probs_test.iloc[:, 1]
    else:
        test_preds[:, fold] = pred_probs_test[:, 1]

# ✅ Test Prediction: 5개 Fold 평균
final_test_preds = test_preds.mean(axis=1)

# ✅ Isotonic Calibration 적용
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(oof_preds, train[label])
calibrated_preds = iso_reg.transform(final_test_preds)


# ✅ 저장 폴더 생성 코드 추가
submission_dir = './submissions'
os.makedirs(submission_dir, exist_ok=True)  # 폴더가 없으면 생성

# ✅ 결과 저장 및 제출
submission_filename = f'{submission_dir}/autogluon_ft_transformer_calibrated.csv'
sample_submission['probability'] = calibrated_preds
sample_submission.to_csv(submission_filename, index=False)

print(f"✅ AutoGluon 학습 완료! 결과가 {submission_filename}에 저장되었습니다.")
