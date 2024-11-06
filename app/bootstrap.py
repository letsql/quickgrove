import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
import os

def enforce_float64(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['int32', 'int64','bool', 'float32', 'float64']).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].astype('float64')
    return df_copy

def prepare_diamonds_data(force_float64: bool = False) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    print("\nPreparing Diamonds Dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
    )

    # this makes `bool` type columns for encoded variables
    df_encoded = pd.get_dummies(
        df,
        columns=["cut", "color", "clarity"],
        prefix={"cut": "cut", "color": "color", "clarity": "clarity"}
    )

    df_encoded.columns = df_encoded.columns.str.replace(" ", "_").str.lower()

    column_order = [
        "carat", "depth", "table", "x", "y", "z",
        "cut_good", "cut_ideal", "cut_premium", "cut_very_good",
        "color_e", "color_f", "color_g", "color_h", "color_i", "color_j",
        "clarity_if", "clarity_si1", "clarity_si2", "clarity_vs1",
        "clarity_vs2", "clarity_vvs1", "clarity_vvs2"
    ]

    for col in column_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0

    X = df_encoded[column_order]
    y = df["price"]

    if force_float64:
        X = enforce_float64(X)
        y = y.astype('float64')

    return X, y, column_order


def prepare_airline_data(force_float64: bool = False) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    print("\nPreparing Airline Dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/varundixit4/Airline-Passenger-Satisfaction-Report/refs/heads/main/airline_satisfaction.csv"
    )
    df = df.drop(["id", "Unnamed: 0"], axis=1)
    df = df.dropna()

    df.columns = df.columns.str.replace(" ", "_").str.lower().str.replace("-", "_")
    
    categorical_columns = [
        "gender", "customer_type", "type_of_travel", 
        "class", "satisfaction"
    ]
    
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    numeric_columns = [
        "age", "flight_distance", "inflight_wifi_service",
        "departure/arrival_time_convenient", "ease_of_online_booking",
        "gate_location", "food_and_drink", "online_boarding",
        "seat_comfort", "inflight_entertainment", "on_board_service",
        "leg_room_service", "baggage_handling", "checkin_service",
        "inflight_service", "cleanliness", "departure_delay_in_minutes",
        "arrival_delay_in_minutes"
    ]
    
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0.0

    column_order = [
        "gender",
        "customer_type",
        "age",
        "type_of_travel",
        "class",
        "flight_distance",
        "inflight_wifi_service",
        "departure/arrival_time_convenient",
        "ease_of_online_booking",
        "gate_location",
        "food_and_drink",
        "online_boarding",
        "seat_comfort",
        "inflight_entertainment",
        "on_board_service",
        "leg_room_service",
        "baggage_handling",
        "checkin_service",
        "inflight_service",
        "cleanliness",
        "departure_delay_in_minutes",
        "arrival_delay_in_minutes",
        
    ]
    
    X = df[column_order]
    y = df["satisfaction"]

    if force_float64:
        X = enforce_float64(X)
        y = y.astype('float64')

    print("\nData Types After Processing:")
    print(X.dtypes)
    return X, y, column_order 

def train_and_save_model(
    name: str, 
    X: pd.DataFrame, 
    y: pd.Series, 
    num_trees: int, 
    force_float64: bool = False,
    learning_rate: float = 0.1
    ) -> Tuple[xgb.Booster, Dict]:
    print(f"\nTraining {name} model...")
    
    if force_float64:
        X = enforce_float64(X)
        y = y.astype('float64')
    
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "max_depth": 6,
        "eta": learning_rate,
        "num_parallel_tree": 10,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "base_score": 0.0
    }

    num_boost_round = num_trees // params["num_parallel_tree"]
    model = xgb.train(params, dtrain, num_boost_round)

    X_full = X.copy()
    X_full["target"] = y
    X_full["prediction"] = model.predict(dtrain)

    if name == "diamonds":
        filtered_data = X_full[X_full["carat"] < 0.3].copy()
        predicate_desc = "carat < 0.3"
    else:
        filtered_data = X_full[X_full["online_boarding"] >= 4.0].copy()
        predicate_desc = "online_boarding >= 4.0"

    os.makedirs("tests/data", exist_ok=True)
    os.makedirs("tests/models", exist_ok=True)

    suffix = "_float64" if force_float64 else ""
    
    X_full.to_csv(f"tests/data/{name}_full{suffix}.csv", index=False)
    filtered_data.to_csv(f"tests/data/{name}_filtered{suffix}.csv", index=False)
    
    model.save_model(f"tests/models/{name}_model{suffix}.json")

    validation_info = {
        "predicate": predicate_desc,
        "model_info": {
            "num_trees": len(model.get_dump()),
            "params": params,
            "num_boost_round": num_boost_round,
            "force_float64": force_float64
        },
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "full_dataset": {
            "row_count": len(X_full),
            "mean_prediction": float(X_full["prediction"].mean()),
            "min_prediction": float(X_full["prediction"].min()),
            "max_prediction": float(X_full["prediction"].max()),
            "predictions_sample": X_full["prediction"].head(5).tolist(),
        },
        "filtered_dataset": {
            "row_count": len(filtered_data),
            "mean_prediction": float(filtered_data["prediction"].mean()),
            "min_prediction": float(filtered_data["prediction"].min()),
            "max_prediction": float(filtered_data["prediction"].max()),
            "predictions_sample": filtered_data["prediction"].head(5).tolist(),
        },
        "feature_names": list(X.columns),
        "feature_stats": {
            col: {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "min": float(X[col].min()),
                "max": float(X[col].max())
            } for col in X.columns
        }
    }

    return model, validation_info

def main():
    force_float64 = False 
    
    X_diamonds, y_diamonds, _ = prepare_diamonds_data(force_float64)
    diamonds_model, diamonds_info = train_and_save_model(
        "diamonds", X_diamonds, y_diamonds, 100, force_float64
    )

    X_airline, y_airline, _ = prepare_airline_data(force_float64)
    airline_model, airline_info = train_and_save_model(
        "airline", X_airline, y_airline, 1500, force_float64
    )

    suffix = " (float64)" if force_float64 else " (original types)"
    print(f"\n=== Model Summary{suffix} ===")
    
    print("\nDiamonds Model:")
    print(f"Number of trees: {len(diamonds_model.get_dump())}")
    print(f"Full dataset rows: {diamonds_info['full_dataset']['row_count']}")
    print(f"Filtered dataset rows: {diamonds_info['filtered_dataset']['row_count']}")
    print("\nFeature Types:")
    for col, dtype in diamonds_info['dtypes'].items():
        print(f"{col}: {dtype}")

    print("\nAirline Model:")
    print(f"Number of trees: {len(airline_model.get_dump())}")
    print(f"Full dataset rows: {airline_info['full_dataset']['row_count']}")
    print(f"Filtered dataset rows: {airline_info['filtered_dataset']['row_count']}")
    print("\nFeature Types:")
    for col, dtype in airline_info['dtypes'].items():
        print(f"{col}: {dtype}")

if __name__ == "__main__":
    main()
