import pandas as pd
from src.data_pipeline import DataWindowManager

df = pd.DataFrame({'a': [1,2,3]})
wm = DataWindowManager(df, ['a'], label_col=None)
print("Is scaler None?", wm.scaler is None)
print("getattr is None?", getattr(wm, "scaler", None) is None)
