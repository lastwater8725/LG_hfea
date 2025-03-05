import os
import pandas as pd
import random
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import KFold
from sklearn.calibration import IsotonicRegression
import warnings

warnings.filterwarnings("ignore")

# ✅ 랜덤 시드 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(69)

# ✅ 데이터 로드
train = pd.read_csv('./data/train.csv').drop(columns=['ID'])
test = pd.read_csv('./data/test.csv').drop(columns=['ID'])
sample_submission = pd.read_csv('./data/sample_submission.csv')

label = '임신 성공 여부'

# ✅ 5-Fold 설정
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ✅ 결과 저장용 배열
oof_preds = np.zeros(len(train))
test_preds = np.zeros((len(test), n_splits))

# ✅ Feature Importance 저장용 리스트
feature_importances = []

# ✅ 5-Fold Cross Validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    train_fold = train.iloc[train_idx]
    valid_fold = train.iloc[valid_idx]

    # ✅ AutoGluon 학습
    predictor = TabularPredictor(
        label=label,
        problem_type='binary',
        eval_metric='roc_auc',
       #path='D:/AutogluonModels/ag-20250223_171750'
    ).fit(
        train_data=TabularDataset(train_fold),
        presets='best_quality',  # 최고 성능 모델 (앙상블 포함)
        time_limit=3600 * 36,  # 10시간 제한
        num_stack_levels=5,  # ✅ Stacking Ensemble (5-Layer)
        num_bag_folds=10,  # ✅ 10-Fold Bagging
        num_bag_sets=2,  # ✅ Bagging 2회 반복 (샘플링 다양화)
        num_gpus=1,  # ✅ GPU 사용
        keep_only_best=True   # ✅ 최적 모델만 저장 (메모리 절약)
    )

    # ✅ Feature Importance 저장
    #feature_importances.append(predictor.feature_importance())

    # ✅ OOF 예측 저장
    pred_probs = predictor.predict_proba(TabularDataset(valid_fold))
    oof_preds[valid_idx] = pred_probs.iloc[:, 1]  # 클래스 1 확률 저장

    # ✅ Test 데이터 예측 저장 (Fold별 평균)
    test_fold_preds = predictor.predict_proba(TabularDataset(test))
    test_preds[:, fold] = test_fold_preds.iloc[:, 1]

# ✅ Test Prediction: 5개 Fold 평균
final_test_preds = test_preds.mean(axis=1)

# ✅ Isotonic Calibration 적용 (OOF 예측 기반)
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(oof_preds, train[label])
calibrated_preds = iso_reg.transform(final_test_preds)

# ✅ 저장 폴더 생성
submission_dir = './submissions'
os.makedirs(submission_dir, exist_ok=True)

# ✅ 결과 저장 및 제출
submission_filename = f'{submission_dir}/autogluon_stacking_ensemble_calibrated.csv'
sample_submission['probability'] = calibrated_preds
sample_submission.to_csv(submission_filename, index=False)

print(f"\n✅ AutoGluon 학습 완료! 결과가 {submission_filename}에 저장되었습니다.")

# # ✅ Feature Importance 저장 (CSV 파일로)
# feature_importance_df = pd.concat(feature_importances).groupby("index").mean().reset_index()
# feature_importance_df.to_csv(f'{submission_dir}/feature_importance.csv', index=False)
# print(f"✅ Feature Importance 저장 완료! (feature_importance.csv)")

# ✅ 리더보드 확인 (Stacking Ensemble 포함)
ld_board = predictor.leaderboard(train, silent=True)
print(ld_board)
