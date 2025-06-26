import pandas as pd

df = pd.read_csv(
    "./logs/predictions.log",
    header=None,
    names=["timestamp", "model", "features", "prediction", "feedback"]
)

df["feedback"] = df["feedback"].astype(str).str.lower() == "true"
df = df[df["feedback"].notnull()]  # Only include rows where feedback was given

results = df.groupby("model")["feedback"].mean()
print("\nModel Accuracy Based on User Feedback:")
print(results)