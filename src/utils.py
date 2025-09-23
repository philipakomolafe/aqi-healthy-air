# NOTE: utils.py contains all the helper functions ike logger for loguru and config loader.
import re
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

config_path = Path(__file__).parent.parent / "config" / "config.yaml"
def config_loader(path=config_path):
    "Helper function to load the config file."
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Convet  from datetime to integer (secs).
def to_unix_timestamp(datetime_obj) -> int:
    return int(time.mktime(datetime_obj.timetuple()))


# TODO: Define the Logger utils to log file processes..
logger_path = Path(__file__).parent.parent / 'logs' / 'pipeline.log'
def setup_logger(log_file: str = logger_path, level: str = "DEBUG"):
    """
    Configure the Loguru logger.
    """
    # Remove existent loggers
    logger.remove()

    # Logging to console.
    logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    # File logging.
    logger.add(log_file, rotation='1 MB', retention="7 days", compression='zip', level=level, enqueue=True, backtrace=True, 
                diagnose=True)

def get_logger(name: str | None = None):
    """
    Create ready to use logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger

def read_processed_data(path: str, log) -> pd.DataFrame:
    """
    Read the CSV file and return a DataFrame.
    """
    try:
        data = pd.read_csv(path)
        log.info(f"Data read successfully from {path}")
        return data
    except FileNotFoundError:
        log.error(f"File not found: {path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        log.error(f"File is empty: {path}")
        return pd.DataFrame()

# TODO: Define the prefect orchestration utilities..

def get_plant_health_info():
    """
    Returns comprehensive plant health information for farmers including disease descriptions and treatments.
    """
    return """
PLANT HEALTH INFORMATION FOR FARMERS:

1. Bacterial Spot (Xanthomonas spp. — primarily X. vesicatoria, X. perforans, X. gardneri, X. euvesicatoria)

DESCRIPTION:
Small, water-soaked spots on leaves turn dark brown to black with yellow halos; spots may merge, causing leaf drop. Caused by Xanthomonas bacteria, spread by water splashes in warm, wet conditions. Gram-negative bacteria spread via infected seeds, transplants, splashing water, tools, and human contact. Thrives in temperatures 24–30°C (75–86°F) with prolonged leaf wetness (>12 hrs).
Symptom Progression:
- Initial: Tiny, water-soaked lesions on leaves, stems, or fruit.
- Mature: Lesions turn dark brown/black, often with chlorotic (yellow) halos.
- Advanced: Lesions coalesce → necrotic patches → defoliation → sunscalded fruit → yield loss.
- Fruit spots: Raised, scabby, brown lesions with cracked centers.

SUGGESTION:
Preventive Measures:
✅ Use certified pathogen-free or hot-water-treated seeds (50°C for 25 min).
✅ Start with disease-free transplants; inspect before planting.
✅ Avoid overhead irrigation — use drip irrigation to minimize leaf wetness.
✅ Space plants for optimal airflow; stake or trellis to elevate foliage.
✅ Copper-based bactericides (e.g., copper hydroxide) applied preventatively every 7–10 days during high-risk periods.
✅ Rotate crops (3–4 years) with non-solanaceous hosts (e.g., corn, beans).
✅ Sanitize tools, gloves, and equipment with 10% bleach or disinfectants.

Corrective Measures:
⚠️ Immediately remove and destroy (burn or deep bury) infected leaves/plants — DO NOT compost.
⚠️ Cease overhead watering; reduce humidity in greenhouse settings.
⚠️ Apply fixed copper + mancozeb tank mix (if labeled) to slow spread — efficacy limited once established.
⚠️ Avoid working in fields when foliage is wet to prevent mechanical spread.
⚠️ Post-harvest: Deep plow debris to accelerate decomposition; solarize soil in warm climates.

—

2. Early Blight (Alternaria solani)
DESCRIPTION:
Dark brown to black spots with concentric rings ("bull's-eye") on lower leaves, often with yellowing. Caused by Alternaria solani, it thrives in warm, wet conditions. Fungal pathogen surviving in soil and plant debris. Spreads via wind, rain, irrigation, tools. Favored by 24–29°C (75–85°F) and >90% RH with 5–10 hrs leaf wetness.
Symptom Progression:
- Starts on oldest leaves: Brown spots with concentric rings ("target spot"), surrounded by yellow chlorosis.
- Lesions enlarge → leaves yellow, wither, drop → defoliation progresses upward.
- Stems: Dark, sunken lesions; fruit: large, leathery, dark spots near stem end.

SUGGESTION:
Preventive Measures:
✅ Plant resistant/tolerant varieties (e.g., 'Mountain Magic', 'Defiant PHR').
✅ Rotate crops (3+ years) away from tomatoes, potatoes, eggplants, peppers.
✅ Mulch with straw or plastic to reduce soil splash.
✅ Prune lower leaves (first 12–18") once plants are established to improve airflow.
✅ Apply protectant fungicides (chlorothalonil, mancozeb) on 7–10 day schedule before symptoms appear.
✅ Avoid nitrogen excess — promotes lush, disease-susceptible foliage.

Corrective Measures:
⚠️ Remove and destroy infected leaves immediately — especially lower canopy.
⚠️ Increase spray frequency to 5–7 days with systemic + protectant fungicides (e.g., azoxystrobin + chlorothalonil).
⚠️ Improve airflow via pruning and wider spacing.
⚠️ Avoid evening irrigation — water early to allow foliage to dry quickly.
⚠️ Post-season: Remove all crop debris; deep till or solarize infested beds.

—

3. Late Blight (Phytophthora infestans)
DESCRIPTION:
Large, irregular gray-green to dark brown spots on leaves, often with white mold in humid conditions. Caused by Phytophthora infestans, spreads rapidly in cool, wet weather. Oomycete (water mold), not a true fungus. Airborne sporangia spread rapidly in cool (10–22°C / 50–72°F), wet (>90% RH, >10 hrs leaf wetness) conditions. Can destroy entire fields in days.

Symptom Progression:
- Leaves: Large, water-soaked, gray-green lesions → turn brown/black; white, fuzzy sporulation on undersides in high humidity.
- Stems: Dark, girdling lesions → plant collapse.
- Fruit: Firm, brown, greasy-looking lesions expanding rapidly.

SUGGESTION:
Preventive Measures:
✅ Monitor local late blight forecasts (e.g., USABlight.org).
✅ Plant resistant varieties (e.g., 'Iron Lady', 'Defiant', 'Mountain Magic' — partial resistance only).
✅ Apply preventative fungicides before conditions favor disease: mancozeb, chlorothalonil, or phosphorous acid products.
✅ Avoid planting near potato fields — same pathogen.
✅ Use drip irrigation; stake plants; space widely for airflow.
✅ Destroy volunteer tomatoes/potatoes and cull piles — major inoculum sources.

Corrective Measures:
⚠️ If detected: Immediately destroy entire infected plants (bag and remove from field — do not compost).
⚠️ Spray remaining plants with systemic fungicides (e.g., mandipropamid, fluopicolide, oxathiapiprolin) + protectant (mancozeb) every 5–7 days.
⚠️ Cease overhead irrigation; reduce humidity in greenhouses with fans/dehumidifiers.
⚠️ Notify neighboring growers — late blight is highly contagious regionally.
⚠️ Post-harvest: Deep plow or solarize; avoid planting solanaceous crops next season.

—

4. Gray Leaf Mold / Leaf Mold (Passalora fulva, formerly Fulvia fulva / Cladosporium fulvum)
DESCRIPTION:
Yellow spots on upper leaf surfaces with grayish-white to olive-green mold on undersides. Caused by Fulvia fulva, favored by high humidity. Strictly foliar fungal pathogen. Spreads via airborne conidia, tools, clothing. Requires high humidity (>85%) and moderate temps (20–25°C / 68–77°F). Common in greenhouses and high tunnels.

Symptom Progression:
- Upper leaf surface: Pale yellow spots → turn brown.
- Underside: Velvety olive-green to gray mold (conidiophores).
- Severe: Leaves curl, dry, drop → reduced photosynthesis → poor fruit set/quality.

SUGGESTION:
Preventive Measures:
✅ Use resistant cultivars (e.g., 'Caruso', 'Trust', 'Geronimo' — check for Fulva race resistance).
✅ Maintain RH <85% — use horizontal airflow (HAF) fans, venting, dehumidifiers.
✅ Water early in day; avoid wetting foliage.
✅ Space plants for airflow; prune suckers and lower leaves.
✅ Apply preventative fungicides: azoxystrobin, difenoconazole, or cyprodinil + fludioxonil.
✅ Sanitize greenhouse structures between crops (bleach, peroxide, or quaternary ammonium).

Corrective Measures:
⚠️ Remove and bag infected leaves immediately — especially lower canopy.
⚠️ Increase ventilation — run fans 24/7 if needed; heat and vent to reduce humidity.
⚠️ Apply curative fungicides: rotate modes of action (FRAC groups) to avoid resistance (e.g., switch from strobilurin to SDHI).
⚠️ Avoid overcrowding — thin plants if necessary.
⚠️ Post-season: Remove all plant debris; sanitize greenhouse benches, walls, tools.
"""

def categorize_aqi(aqi: float):
    if aqi <= 0:
        return "Invalid Air Quality Index (AQI) value. Please check the data source."
    elif aqi <= 1:
        return "Excellent: Safe for outdoor activities, but stay alert to symptoms."
    elif aqi <= 2:
        return "Good: Generally safe, but consider wearing a mask if outdoors for long periods."
    elif aqi <= 3:
        return "Moderate: Limit prolonged outdoor exertion. Carry inhaler/medication. Consider a mask."
    elif aqi <= 4:
        return "Poor: Avoid outdoor activities. Use air purifiers indoors. Wear a mask if you must go out."
    elif aqi <= 5:
        return "Very Poor: Stay indoors with windows closed. Use HEPA filters. Avoid physical activity outside."
    else:
        return "Hazardous: Remain indoors. Use air purifiers. Seek medical attention if symptoms worsen." 

def categorize_aqi_with_plant_health(aqi: float):
    """
    Categorize AQI and provide plant health recommendations for farmers.
    """
    plant_health_info = get_plant_health_info()
    
    if aqi <= 0:
        base_advice = "Invalid Air Quality Index (AQI) value. Please check the data source."
        plant_advice = "Cannot provide plant health recommendations for invalid AQI values."
    elif aqi <= 1:
        base_advice = "Excellent: Safe for outdoor activities, but stay alert to symptoms."
        plant_advice = ("PLANT HEALTH (Excellent Air Quality): Optimal conditions for plant growth. "
                       "Perfect time for planting, harvesting, and general farm work. "
                       "Regular plant health monitoring and preventive care recommended.")
    elif aqi <= 2:
        base_advice = "Good: Generally safe, but consider wearing a mask if outdoors for long periods."
        plant_advice = ("PLANT HEALTH (Good Air Quality): Good conditions for most farming activities. "
                       "Ideal for disease prevention measures and plant health maintenance. "
                       "Monitor for early signs of plant diseases.")
    elif aqi <= 3:
        base_advice = "Moderate: Limit prolonged outdoor exertion. Carry inhaler/medication. Consider a mask."
        plant_advice = ("PLANT HEALTH (Moderate Air Quality): Increased risk of plant stress. "
                       "Enhanced monitoring needed for bacterial spot and early blight. "
                       "Implement preventive fungicide applications. "
                       "Focus on improving plant airflow and reducing moisture.")
    elif aqi <= 4:
        base_advice = "Poor: Avoid outdoor activities. Use air purifiers indoors. Wear a mask if you must go out."
        plant_advice = ("PLANT HEALTH (Poor Air Quality): HIGH RISK for plant diseases. "
                       "Monitor closely for late blight and gray leaf mold. "
                       "Increase fungicide application frequency. "
                       "Improve greenhouse ventilation. Avoid overhead irrigation.")
    elif aqi <= 5:
        base_advice = "Very Poor: Stay indoors with windows closed. Use HEPA filters. Avoid physical activity outside."
        plant_advice = ("PLANT HEALTH (Very Poor Air Quality): CRITICAL - Severe plant disease risk. "
                       "Emergency plant protection measures required. "
                       "Maximum ventilation in greenhouses. "
                       "Daily disease monitoring. Remove infected plants immediately.")
    else:
        base_advice = "Hazardous: Remain indoors. Use air purifiers. Seek medical attention if symptoms worsen."
        plant_advice = ("PLANT HEALTH (Hazardous Air Quality): EXTREME DANGER for crops. "
                       "Suspend outdoor farming activities if possible. "
                       "Protect greenhouse crops with maximum air filtration. "
                       "Emergency harvest of healthy crops recommended.")
    
    return f"{base_advice}\n\n{plant_advice}\n\n{plant_health_info}"

def color_aqi(aqi: float):
    if aqi <= 0:
        return "Invalid"
    elif aqi <= 1:
        return "Green"
    elif aqi <= 2:
        return "Yellow"
    elif aqi <= 3:
        return "Orange"
    elif aqi <= 4:
        return "Red"
    elif aqi <= 5:
        return "Purple"
    else:
        return "Pale"
    if aqi <= 0:
        return "Invalid"
    elif aqi <= 1:
        return "Green"
    elif aqi <= 2:
        return "Yellow"
    elif aqi <= 3:
        return "Orange"
    elif aqi <= 4:
        return "Red"
    elif aqi <= 5:
        return "Purple"
    else:
        return "Pale"

def postprocess_predictions(predictions: np.ndarray, include_plant_health: bool = False) -> pd.DataFrame:
    """
    postprocess the predictions for health related user-friendly AQI values.
    
    Args:
        predictions: Array of AQI predictions
        include_plant_health: If True, includes comprehensive plant health information for farmers
    
    Returns:
        DataFrame with AQI categories, colors, and optionally plant health information
    """
    df = pd.DataFrame({"predicted_aqi": predictions})
    
    if include_plant_health:
        # categorize the AQI level with plant health information for farmers
        df['aqi_category'] = df['predicted_aqi'].apply(categorize_aqi_with_plant_health)
    else:
        # categorize the AQI level as good, bad or e.t.c for health conscious population.
        df['aqi_category'] = df['predicted_aqi'].apply(categorize_aqi)
    
    # Air quality health color to give patient deeper view on the AQI status..
    df['aqi_health_color'] = df['predicted_aqi'].apply(color_aqi) 
    return df

def fetch_current_data(config):
    try:
        start_ts = int(time.mktime(datetime.today().timetuple()))
        end_ts = int(time.mktime((datetime.today() + timedelta(hours=1)).timetuple()))
        params=config['openweather']['params']
        params['start']=start_ts
        params['end']=end_ts

        response = requests.get(
            url=config['openweather']['base_url'],
            params=params
        )
        # Similarly checking for HTTPerror. Totology anyways.
        response.raise_for_status()
        # Dump data into JSON format.
        data = response.json()

        # Return the features to compute the AQI value.
        return {
                'no': data['list'][0]['components']['no'],
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'co': data['list'][0]['components']['co'],
                'no2': data['list'][0]['components']['no2'],
                'so2': data['list'][0]['components']['so2'],
                'o3': data['list'][0]['components']['o3'],
                'nh3': data['list'][0]['components']['nh3'],
                'timestamp': datetime.utcfromtimestamp(data['list'][0]['dt']),
                'aqi': data['list'][0]['main']['aqi']
                }
                # 'timestamp', 'aqi', 'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3', 'nh3'
    
    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {e}"
    except Exception as e:
        return f"Other Errors occurred -> {e}"


def select_best_model(model_directory: Path, weight=(0.6, 0.4)) -> tuple[str, float, float, float, str]:
    """
    Select the best model based off the weighted sum of the Balanced accuracy and Macro ROC-AUC values.
    Handles both classical ML models (.pkl) and deep learning models (directories with keras_model.h5).
    """
    best_model_path = None
    best_roc = -1
    best_acc = -1
    best_score = -1
    best_model_type = None

    # Define search patterns for both classical and deep learning models
    classical_pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)\.pkl")
    dl_pattern = re.compile(r"(\w+)_acc_(\d+\.\d+)_roc_(\d+\.\d+)")

    # Iterate through all files and directories
    for item in os.listdir(model_directory):
        item_path = os.path.join(model_directory, item)
        
        if os.path.isfile(item_path) and item.endswith('.pkl'):
            # Classical ML model
            match = classical_pattern.search(item)
            if match:
                model_name = match.group(1)
                acc = float(match.group(2))
                roc = float(match.group(3))
                
                # Weighted sum of balanced accuracy and macro ROC
                w1, w2 = weight
                score = (w1 * roc) + (w2 * acc)
                
                if score > best_score:
                    best_score = score
                    best_model_path = item_path
                    best_acc = acc
                    best_roc = roc
                    best_model_type = 'classical'
                    
        elif os.path.isdir(item_path):
            # Check if it's a deep learning model directory
            keras_model_path = os.path.join(item_path, 'keras_model.h5')
            metadata_path = os.path.join(item_path, 'metadata.pkl')
            
            if os.path.exists(keras_model_path) and os.path.exists(metadata_path):
                # Deep learning model directory
                match = dl_pattern.search(item)
                if match:
                    model_name = match.group(1)
                    acc = float(match.group(2))
                    roc = float(match.group(3))
                    
                    # Weighted sum of balanced accuracy and macro ROC
                    w1, w2 = weight
                    score = (w1 * roc) + (w2 * acc)
                    
                    if score > best_score:
                        best_score = score
                        best_model_path = item_path  # Return directory path for DL models
                        best_acc = acc
                        best_roc = roc
                        best_model_type = 'deep_learning'

    return best_model_path, best_acc, best_roc, best_score, best_model_type


if __name__ == "__main__":
    pass
