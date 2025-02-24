import pandas as pd
import numpy as np
import glob

# 🔥 제출 파일이 저장된 폴더 경로 (수동 수정)
submission_dir = "C:/Users/okpl8/Documents/project_LG/submissions"


# 🔥 제출 파일별 가중치 설정 (성능에 따라 직접 조정)
weights = {
    f"{submission_dir}/autogluon_ft_transformer_calibrated.csv": 0.7,
    f"{submission_dir}/submit5.csv": 0.3,
}

# 파일 로드
submissions = {fname: pd.read_csv(fname) for fname in weights.keys()}

# 가중 평균 계산
weighted_preds = np.zeros(len(next(iter(submissions.values()))))

for fname, df in submissions.items():
    weighted_preds += weights[fname] * df["probability"]

# 최종 결과 저장
final_submission = next(iter(submissions.values())).copy()
final_submission["probability"] = weighted_preds
output_path = f"{submission_dir}/weighted_ensemble_submission.csv"
final_submission.to_csv(output_path, index=False)

print(f"✅ 가중 평균 앙상블 완료! 결과 저장: {output_path}")
