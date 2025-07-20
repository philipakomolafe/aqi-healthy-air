import sweetviz as sv
import pandas as pd
import os

# Load your dataset
df = pd.read_csv("data/processed/train/aqi_train_data_v1.csv")

# Generate the report
report = sv.analyze(df)

# Save the report
os.makedirs('eda', exists_ok=True)
report.show_html("eda/eda_report.html")
