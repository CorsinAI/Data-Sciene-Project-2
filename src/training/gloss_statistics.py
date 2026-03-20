from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# CSV_PATH = PROJECT_ROOT / "data" / "processed" / "wlasl_model_metadata_min_frames_8.csv"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "custom_metadata.csv"

df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))
print("Total glosses:", df["gloss"].nunique())

print("\nTop 20 glosses:")
print(df["gloss"].value_counts().head(20))

print("\nTop 10 split table:")
top10 = df["gloss"].value_counts().head(10).index
df_top10 = df[df["gloss"].isin(top10)]
print(pd.crosstab(df_top10["gloss"], df_top10["split"]))