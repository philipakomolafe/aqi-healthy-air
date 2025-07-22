import sweetviz as sv
import pandas as pd
import os
from pathlib import Path

# Define the working dir.
root = Path(__file__).parent

def show_eda():
    # Load your dataset
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'train', 'aqi_train_data_v1.csv'))

    # Generate the report
    report = sv.analyze(df)

    # Save the report
    # os.makedirs('eda', exists_ok=True)
    report.show_html("eda_report.html")

if __name__ == "__main__":
    # Output the EDA of the data..
    show_eda()
    # pass