# Air Quality Index (AQI) Machine Learning Project

This project provides a modular, production-ready framework for working with Air Quality Index (AQI) data. It includes three main pipelines:

- **Feature Pipeline**: For data cleaning, feature engineering, feature selection, and data splitting.
- **Training Pipeline**: For model training and evaluation (to be implemented).
- **Inference Pipeline**: For making predictions with trained models (to be implemented).

The project is designed for extensibility, reproducibility, and ease of use, leveraging best practices in data science and software engineering.

## Project Structure

```
├── config/
│   └── config.yaml           # Configuration file for API, data paths, etc.
├── data/
│   ├── raw/                  # Raw AQI data (CSV)
│   ├── processed/            # Processed and feature-engineered data
│   └── cache/                # Cached intermediate results
├── logs/
│   └── pipeline.log          # Log file for pipeline runs
├── models/                   # Model storage
├── pipelines/
│   └── feature_pipeline.py   # Main feature pipeline script
├── src/
│   ├── data_cleaner.py       # Data cleaning utilities
│   ├── data_fetcher.py       # (Optional) Data fetching utilities
│   ├── data_splitter.py      # Time-based data splitting
│   ├── feature_cache.py      # Caching utilities
│   ├── feature_engineering.py# Feature engineering for AQI
│   ├── feature_selection.py  # Feature selection (correlation, importance)
│   ├── feature_storage.py    # Feature saving/versioning
│   ├── neptune_utils.py      # Neptune logging utilities
│   └── utils.py              # Config loader, logger setup, helpers
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Features

- **Configurable**: All paths and API keys are managed in `config/config.yaml`.
- **Logging**: Uses Loguru for robust logging to `logs/pipeline.log`.
- **Modular**: Each pipeline step (cleaning, engineering, selection, splitting, saving, caching) is a separate script in `src/`.
- **Feature Engineering**: Adds time-based, weather, interaction, and AQI-specific features.
- **Feature Selection**: Removes highly correlated features and ranks features by importance.
- **Time Series Splitting**: Splits data into train/validation/test sets chronologically.
- **Feature Storage & Caching**: Saves processed features with versioning and supports caching for speed.

## How to Use

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Your Project**
   - Edit `config/config.yaml` to set your data paths and API keys.
   - Sensitive info (API keys) should be stored in a `.env` file and referenced in your config.

3. **Run the Feature Pipeline**
   ```bash
   python pipelines/feature_pipeline.py
   ```
   This will:
   - Load and clean raw AQI data
   - Engineer new features
   - Select the best features
   - Split data into train/val/test sets
   - Save processed features to `data/processed/`
   - Log all steps to `logs/pipeline.log`

4. **Check Outputs**
   - Processed feature files: `data/processed/aqi_train_data_*.csv`, etc.
   - Logs: `logs/pipeline.log`

## Customization
- Add new feature engineering logic in `src/feature_engineering.py`.
- Adjust feature selection thresholds in `src/feature_selection.py`.
- Change data split ratios in `src/data_splitter.py`.
- Update logging or config logic in `src/utils.py`.

## Best Practices
- Keep sensitive info out of version control (add `.env` and `config/config.yaml` to `.gitignore`).
- Use the logger for all print/debug statements.
- Version your features for reproducibility.

## Credits
- Built with Loguru, Pandas, Scikit-learn, and Neptune.ai (for experiment tracking).

---

**This pipeline is designed for easy extension and robust, reproducible AQI data science workflows.**
