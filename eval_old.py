from predict import _get_stage1_model, _get_stage2_model
from src.data_loader import build_dataframe_from_csv, get_train_val_split
from src.cascade_train import evaluate_cascade

df = build_dataframe_from_csv()
_, val_df = get_train_val_split(df)
s1 = _get_stage1_model()
s2 = _get_stage2_model()
for t in (0.6, 0.35):
    evaluate_cascade(s1, s2, val_df, stage1_threshold=t)