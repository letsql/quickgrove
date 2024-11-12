from typing import Optional, Literal, Dict, Any, Tuple, List, Type
from pathlib import Path
import argparse
import attrs
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import json
from abc import ABC, abstractmethod
from enum import Enum

pd.options.mode.copy_on_write = True

class DataVariant(str, Enum):
    FULL = "full"
    FILTERED = "filtered"

class GenerationType(str, Enum):
    BENCHMARK = "benchmark"
    TEST = "test"

class ObjectiveType(str, Enum):
    SQUARED_ERROR = "reg:squarederror"
    LOGISTIC = "reg:logistic"
    BINARY_LOGISTIC = "binary:logistic"

@attrs.define(frozen=True)
class DataConfig:
    name: Literal["diamonds", "airline_satisfaction"]
    generation_type: GenerationType
    variant: DataVariant
    data_dir: Path 
    force_float64: bool = False

    @property
    def sample_size(self) -> Optional[int]:
        return 100 if self.generation_type == GenerationType.TEST else None

    @property
    def filter_predicate(self) -> Optional[str]:
        if self.variant == DataVariant.FILTERED:
            predicates = {
                "diamonds": "carat < 0.3",
                "airline_satisfaction": "online_boarding >= 4.0"
            }
            return predicates.get(self.name)
        return None

    @property
    def data_path(self) -> Path:
        return self.data_dir / f"{self.name}.csv"

@attrs.define(frozen=True)
class TreeConfig:
    num_trees: int
    allowed_datasets: List[str]

    @classmethod
    def get_configs(cls, generation_type: GenerationType) -> List["TreeConfig"]:
        if generation_type == GenerationType.TEST:
            return [cls(num_trees=10, allowed_datasets=["diamonds", "airline_satisfaction"])]
        return [
            cls(num_trees=100, allowed_datasets=["diamonds", "airline_satisfaction"]),
            cls(num_trees=500, allowed_datasets=["airline_satisfaction"]),
            cls(num_trees=1000, allowed_datasets=["airline_satisfaction"])
        ]

@attrs.define(frozen=True)
class ObjectiveConfig:
    name: ObjectiveType
    generation_type: GenerationType
    num_trees: int

    @property
    def num_parallel_trees(self) -> int:
        return 10

    @property
    def num_boost_rounds(self) -> int:
        return self.num_trees // self.num_parallel_trees

    @property
    def learning_rate(self) -> float:
        return 0.1

    @property
    def max_depth(self) -> int:
        return 6

    def to_xgb_params(self, base_score: Optional[float] = None) -> Dict[str, Any]:
        params = {
            "objective": self.name.value,
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "num_parallel_tree": self.num_parallel_trees,
            "eval_metric": "rmse" if self.name.value.startswith("reg:") else "logloss"
        }
        if base_score is not None:
            params["base_score"] = base_score
        return params

@attrs.define
class ModelMetadata:
    dataset_name: str
    objective_name: str
    variant: str
    force_float64: bool
    feature_names: List[str]
    data_shape: Tuple[int, int]
    num_trees: int

@attrs.define
class OutputPaths:
    base_dir: Path
    dataset_name: str
    variant: str
    objective_name: str
    force_float64: bool
    generation_type: GenerationType = GenerationType.BENCHMARK
    num_trees: Optional[int] = None

    @property
    def data_suffix(self) -> str:
        return f"{self.variant}_{'float64' if self.force_float64 else 'mixed'}"

    @property
    def model_suffix(self) -> str:
        return f"trees_{self.num_trees}_{'float64' if self.force_float64 else 'mixed'}"

    @property
    def output_base_dir(self) -> Path:
        subdir = "tests" if "test" in str(self.generation_type) else "benches"
        return self.base_dir / subdir

    @property
    def data_path(self) -> Path:
        return (self.output_base_dir / self.objective_name / "data" /
                f"{self.dataset_name}_data_{self.data_suffix}.csv")

    @property
    def model_path(self) -> Path:
        if self.num_trees is None:
            raise ValueError("num_trees must be set for model path")
        return (self.output_base_dir / self.objective_name / "models" /
                f"{self.dataset_name}_model_{self.model_suffix}.json")

    @property
    def metadata_path(self) -> Path:
        if self.num_trees is None:
            raise ValueError("num_trees must be set for metadata path")
        return (self.output_base_dir /  self.objective_name / "models" /
                f"{self.dataset_name}_metadata_{self.model_suffix}.json")

    def ensure_directories(self) -> None:
        """Create necessary directories for all paths."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

class DataProcessor(ABC):
    def __init__(self, config: DataConfig):
        self.config = config

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw data from source."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data."""
        pass

    @abstractmethod
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        pass

    def enforce_float64(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(
            include=['int32', 'int64', 'bool', 'float32', 'float64']
        ).columns
        for col in numeric_cols:
            df_copy[col] = df_copy[col].astype('float64')
        return df_copy

class DiamondsProcessor(DataProcessor):
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.column_order = [
            "carat", "depth", "table", "x", "y", "z",
            "cut_good", "cut_ideal", "cut_premium", "cut_very_good",
            "color_e", "color_f", "color_g", "color_h", "color_i", "color_j",
            "clarity_if", "clarity_si1", "clarity_si2", "clarity_vs1",
            "clarity_vs2", "clarity_vvs1", "clarity_vvs2"
        ]

    def load_data(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        return pd.read_csv(self.config.data_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # if self.config.filter_predicate:
        #     df = df.query(self.config.filter_predicate)

        if self.config.sample_size:
            df = df.sample(n=min(self.config.sample_size, len(df)), random_state=42)

        df_encoded = pd.get_dummies(
            df,
            columns=["cut", "color", "clarity"],
            prefix={"cut": "cut", "color": "color", "clarity": "clarity"}
        )
        df_encoded.columns = df_encoded.columns.str.replace(" ", "_").str.lower()

        for col in self.column_order:
            if col not in df_encoded.columns:
                df_encoded[col] = False
            
            if col.startswith(("cut_", "color_", "clarity_")):
                df_encoded[col] = df_encoded[col].astype(bool)

        numeric_cols = ["carat", "depth", "table", "x", "y", "z"]
        for col in numeric_cols:
            df_encoded[col] = df_encoded[col].astype('float64')

        return df_encoded[self.column_order + ["price"]]

    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[self.column_order]
        y = df["price"]

        if self.config.force_float64:
            X = self.enforce_float64(X)
            y = y.astype('float64')

        return X, y

class AirlineProcessor(DataProcessor):
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.column_order = [
            "gender", "customer_type", "age", "type_of_travel", "class",
            "flight_distance", "inflight_wifi_service",
            "departure/arrival_time_convenient", "ease_of_online_booking",
            "gate_location", "food_and_drink", "online_boarding",
            "seat_comfort", "inflight_entertainment", "on_board_service",
            "leg_room_service", "baggage_handling", "checkin_service",
            "inflight_service", "cleanliness", "departure_delay_in_minutes",
            "arrival_delay_in_minutes"
        ]

    def load_data(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        return df.drop(["id", "Unnamed: 0"], axis=1) if "id" in df.columns else df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df.columns = df.columns.str.replace(" ", "_").str.lower().str.replace("-", "_")

        # if self.config.filter_predicate:
        #     df = df.query(self.config.filter_predicate)

        if self.config.sample_size:
            df = df.sample(n=min(self.config.sample_size, len(df)), random_state=42)

        categorical_columns = [
            "gender", "customer_type", "type_of_travel",
            "class", "satisfaction"
        ]

        le = LabelEncoder()
        for col in categorical_columns:
            if col in df.columns:
                _t = le.fit_transform(df[col])
                df[col] = _t.copy()

        return df

    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[self.column_order]
        y = df["satisfaction"]

        if self.config.force_float64:
            X = self.enforce_float64(X)
            y = y.astype('float64')

        return X, y

class ModelTrainer:
    def __init__(
        self,
        data_processor: DataProcessor,
        objective_config: ObjectiveConfig,
        base_dir: Path
    ):
        self.data_processor = data_processor
        self.objective_config = objective_config
        self.base_dir = base_dir

    def _get_paths(self, data_config: DataConfig) -> OutputPaths:
        return OutputPaths(
            base_dir=self.base_dir,
            dataset_name=data_config.name,
            variant=data_config.variant.value,
            objective_name=self.objective_config.name.value,
            force_float64=data_config.force_float64,
            generation_type=data_config.generation_type,
            num_trees=self.objective_config.num_trees
        )

    def prepare_data_for_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X_prep = X.copy()
        y_prep = y.copy()

        if self.objective_config.name in [ObjectiveType.LOGISTIC, ObjectiveType.BINARY_LOGISTIC]:
            y_prep = (y_prep - y_prep.min()) / (y_prep.max() - y_prep.min())

        return X_prep, y_prep

    def train_and_save(self, data_config: DataConfig) -> None:
        paths = self._get_paths(data_config)
        paths.ensure_directories()

        df = self.data_processor.load_data()
        df_full = self.data_processor.preprocess(df.copy())
        X_full, y_full = self.data_processor.get_feature_target_split(df_full)
        
        X_prep_full, y_prep_full = self.prepare_data_for_objective(X_full, y_full)
        dtrain_full = xgb.DMatrix(X_prep_full, label=y_prep_full)
        
        base_score = float(y_prep_full.mean()) if self.objective_config.name in [
            ObjectiveType.LOGISTIC, ObjectiveType.BINARY_LOGISTIC
        ] else None
        
        params = self.objective_config.to_xgb_params(base_score)
        model = xgb.train(params, dtrain_full, self.objective_config.num_boost_rounds)

        df_preprocessed = self.data_processor.preprocess(df)
        X, y = self.data_processor.get_feature_target_split(df_preprocessed)
        X_prep, y_prep = self.prepare_data_for_objective(X, y)
        
        dtrain = xgb.DMatrix(X_prep)
        predictions = model.predict(dtrain)

        output_data = X_prep.copy()
        output_data['target'] = y_prep.astype('int64')
        output_data['prediction'] = predictions.astype('float64')
        if data_config.filter_predicate:
            output_data = output_data.query(data_config.filter_predicate)
        output_data.to_csv(paths.data_path, index=False)

        model.save_model(str(paths.model_path))
        print(f""" Written files to:
        * {paths.data_path}
        * {paths.model_path}""")
        
        metadata = ModelMetadata(
            dataset_name=data_config.name,
            objective_name=self.objective_config.name.value,
            variant=data_config.variant.value,
            force_float64=data_config.force_float64,
            feature_names=list(X_prep.columns),
            data_shape=X_prep.shape,
            num_trees=len(model.get_dump())
        )

        with open(paths.metadata_path, 'w') as f:
            json.dump(attrs.asdict(metadata), f, indent=2)

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--generation_type", type=str, default="benchmark")
    parser.add_argument("--base_dir", type=str, default=".")
    return parser.parse_args()

def main():
    args = arg_parse()
    processors: Dict[str, Type[DataProcessor]] = {
        "diamonds": DiamondsProcessor,
        "airline_satisfaction": AirlineProcessor
    }

    generation_type = args.generation_type
    data_dir = Path(args.data_dir)
    base_dir = Path(args.base_dir)
    
    for objective_name in [ObjectiveType.SQUARED_ERROR]:
        for dataset_name, processor_cls in processors.items():
            for force_float64 in [False, True]:
                for variant in DataVariant:
                    for tree_config in TreeConfig.get_configs(generation_type):
                        if dataset_name not in tree_config.allowed_datasets:
                            continue

                        try:
                            data_config = DataConfig(
                                name=dataset_name,
                                generation_type=generation_type,
                                variant=variant,
                                data_dir=data_dir,
                                force_float64=force_float64
                            )

                            objective_config = ObjectiveConfig(
                                name=objective_name,
                                generation_type=generation_type,
                                num_trees=tree_config.num_trees
                            )

                            processor = processor_cls(data_config)
                            trainer = ModelTrainer(
                                processor,
                                objective_config,
                                base_dir
                            )

                            trainer.train_and_save(data_config)
                            print(f"✨Successfully processed {dataset_name} dataset with {tree_config.num_trees} trees\n")

                        except Exception as e:
                            print(f"Error processing {dataset_name} dataset: {str(e)}")
                            continue

if __name__ == "__main__":
    main()
