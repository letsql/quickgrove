import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def prepare_diamonds_data():
    print("\nPreparing Diamonds Dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
    )

    df_encoded = pd.get_dummies(
        df,
        columns=["cut", "color", "clarity"],
        prefix={"cut": "cut", "color": "color", "clarity": "clarity"},
        dtype="int64",
    )

    df_encoded.columns = df_encoded.columns.str.replace(" ", "_").str.lower()

    column_order = [
        "carat",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "cut_good",
        "cut_ideal",
        "cut_premium",
        "cut_very_good",
        "color_e",
        "color_f",
        "color_g",
        "color_h",
        "color_i",
        "color_j",
        "clarity_if",
        "clarity_si1",
        "clarity_si2",
        "clarity_vs1",
        "clarity_vs2",
        "clarity_vvs1",
        "clarity_vvs2",
    ]

    for col in column_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[column_order]
    y = df["price"].astype("float64")

    return X, y, column_order


def prepare_airline_data():
    print("\nPreparing Airline Dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/varundixit4/Airline-Passenger-Satisfaction-Report/refs/heads/main/airline_satisfaction.csv"
    )
    df = df.drop(["id", "Unnamed: 0"], axis=1)
    df = df.dropna()

    df.columns = df.columns.str.replace(" ", "_").str.lower().str.replace("-", "_")
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

    for col in column_order:
        if col not in df.columns:
            df[col] = 0

    X = df[column_order]
    y = df["satisfaction"]

    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    print(X.dtypes)
    return X, y, column_order


def train_and_save_model(name, X, y, num_trees, learning_rate=0.1):
    print(f"\nTraining {name} model...")
    dtrain = xgb.DMatrix(X, label=y)

    params = {"max_depth": 6, "eta": learning_rate, "num_parallel_tree": 10}

    if name == "airline":
        params.update(
            {
                # 'objective': 'binary:logistic',
                # 'eval_metric': 'logloss'
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
        )
    else:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    num_boost_round = num_trees // params["num_parallel_tree"]
    model = xgb.train(params, dtrain, num_boost_round)

    X_full = X.copy()
    X_full["target"] = y
    X_full["prediction"] = model.predict(dtrain)

    if name == "diamonds":
        filtered_data = X_full[X_full["carat"] <= 0.3].copy()
        predicate_desc = "carat <= 0.3"
    else:
        filtered_data = X_full[X_full["online_boarding"] > 4].copy()
        predicate_desc = "online_boarding > 4"

    X_full.to_csv(f"tests/data/{name}_full.csv", index=False)
    filtered_data.to_csv(f"tests/data/{name}_filtered.csv", index=False)

    model.save_model(f"tests/models/{name}_model.json")
    validation_info = {
        "predicate": predicate_desc,
        "model_info": {
            "num_trees": len(model.get_dump()),
            "params": params,
            "num_boost_round": num_boost_round,
        },
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
    }

    return model, validation_info


def main():
    X_diamonds, y_diamonds, _ = prepare_diamonds_data()
    diamonds_model, diamonds_info = train_and_save_model(
        "diamonds", X_diamonds, y_diamonds, 100
    )

    X_airline, y_airline, _ = prepare_airline_data()
    airline_model, airline_info = train_and_save_model(
        "airline", X_airline, y_airline, 1500
    )

    print("\n=== Model Summary ===")

    print("\nDiamonds Model:")
    print(f"Number of trees: {len(diamonds_model.get_dump())}")
    print(f"Full dataset rows: {diamonds_info['full_dataset']['row_count']}")
    print(f"Filtered dataset rows: {diamonds_info['filtered_dataset']['row_count']}")
    print(f"Features: {diamonds_info['feature_names']}")

    print("\nAirline Model:")
    print(f"Number of trees: {len(airline_model.get_dump())}")
    print(f"Full dataset rows: {airline_info['full_dataset']['row_count']}")
    print(f"Filtered dataset rows: {airline_info['filtered_dataset']['row_count']}")
    print(f"Features: {airline_info['feature_names']}")


if __name__ == "__main__":
    main()
