# TODO: this checks the data's model training usefulness.
import os
import pandas as pd
import numpy as np
from .utils import config_loader, setup_logger, get_logger

# Init logger configuration.
setup_logger(level="INFO")
log = get_logger()

# Init Config.
config = config_loader()

# This function helps load the `OPENWEATHER API DATA`. 
def load_data(data_path: str = config['dataset']['raw']) -> pd.DataFrame : 
    # Ensure data_path is relative to project root
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)
    if os.path.exists(data_path):
        filename = "aqi_data_2016-2025.csv"
        path = os.path.join(data_path, filename)
        return pd.read_csv(path)
    else:
        print(f"Path `{data_path}` provided doesn't exist...")
        os.makedirs(data_path, exist_ok=True)
        return pd.read_csv(data_path)
    

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the AQI dataset.
    
    Args:
        data (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed data
    """
    try:
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # 1. Handle missing values
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # 2. Remove duplicates
        df = df.drop_duplicates()
        
        # 3. Handle outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 4. Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                log.debug(f"Could not convert {col} to datetime")
        
        # 5. Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # 6. Remove any leading/trailing whitespace in string columns
        df.columns = df.columns.str.strip()
        # for col in categorical_cols:
        #     df[col] = df[col].str.strip()
        
        # 7. Log data quality report
        log.info(f"\nData Quality Report:  \nTotal rows: {len(df)}\nTotal columns: {len(df.columns)}")
        log.info(f"\nMissing values after cleaning:  \n{df.isnull().sum()}")
        log.info(f"\nData types: \n{df.dtypes}")
        
        return df
        
    except Exception as e:
        log.error(f"Error during data cleaning: {str(e)}")
        raise


if __name__ == "__main__":
    # Load the api fetched data.
    raw = load_data()
    clean = clean_data(raw)
    log.info("\nSample of cleaned data:")
    log.info(clean.head())