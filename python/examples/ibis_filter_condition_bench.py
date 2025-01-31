import functools
import time
from typing import List, Tuple

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ibis import _
from ibis.common.annotations import Argument
from ibis.common.collections import FrozenDict
from ibis.common.patterns import pattern, replace
from ibis.expr.operations.udf import InputType, ScalarUDF
from ibis.expr.rules import ValueOf
from ibis.util import Namespace
from quickgrove import Feature, PyGradientBoostedDecisionTrees


def write_large_parquet(
    input_csv, output_parquet, target_rows=25_000_000, chunk_size=500_000
):
    df = pd.read_csv(input_csv)
    rows_per_chunk = min(chunk_size, len(df))

    # Create schema from first chunk
    chunk = pd.concat([df] * (chunk_size // len(df) + 1), ignore_index=True)[
        :rows_per_chunk
    ]
    table = pa.Table.from_pandas(chunk)
    writer = pq.ParquetWriter(output_parquet, table.schema)

    # Write chunks
    rows_written = 0
    while rows_written < target_rows:
        chunk = pd.concat([df] * (chunk_size // len(df) + 1), ignore_index=True)[
            :rows_per_chunk
        ]
        table = pa.Table.from_pandas(chunk)
        writer.write_table(table)
        rows_written += rows_per_chunk
        print(f"Written {rows_written:,} rows")

    writer.close()


p = Namespace(pattern, module=ops)


def collect_predicates(filter_op: ops.Filter) -> List[dict]:
    """Extract predicates from a Filter operation."""
    predicates = []
    for pred in filter_op.predicates:
        if isinstance(pred, ops.Less):
            if isinstance(pred.left, ops.Field):
                predicates.append(
                    {
                        "column": pred.left.name,
                        "op": "Less",
                        "value": pred.right.value
                        if isinstance(pred.right, ops.Literal)
                        else None,
                    }
                )
    return predicates


def create_pruned_udf(
    original_udf: callable,
    model: PyGradientBoostedDecisionTrees,
    predicates: List[dict],
) -> Tuple[callable, List[str]]:
    """Create a new UDF using the pruned model based on predicates."""

    pruned_model = model.prune(
        [
            Feature(pred["column"]) < pred["value"]
            for pred in predicates
            if pred["op"] == "Less" and pred["value"] is not None
        ]
    )

    required_features = sorted(pruned_model.required_features)
    feature_names = [model.feature_names[i] for i in required_features]

    def fn_from_arrays(*arrays):
        return pruned_model.predict_arrays(list(arrays))

    fields = {
        feature_name: Argument(pattern=ValueOf(dt.float64), typehint=dt.float64)
        for feature_name in feature_names
    }

    meta = {
        "dtype": dt.float32,
        "__input_type__": InputType.PYARROW,
        "__func__": property(lambda self: fn_from_arrays),
        "__config__": FrozenDict(volatility="immutable"),
        "__udf_namespace__": Namespace(pattern, module=ops),
        "__module__": original_udf.__module__,
        "__func_name__": original_udf.__name__ + "_pruned",
    }

    node = type(original_udf.__name__ + "_pruned", (ScalarUDF,), {**fields, **meta})

    @functools.wraps(fn_from_arrays)
    def construct(*args, **kwargs):
        return node(*args, **kwargs).to_expr()

    construct.fn = fn_from_arrays

    return construct, feature_names


@replace(p.Filter)
def prune_gbdt_model(_, **kwargs):
    """Rewrite rule to prune GBDT model based on filter predicates."""

    model = kwargs["model"]
    original_udf = kwargs["original_udf"]

    predicates = collect_predicates(_)
    if not predicates:
        return _

    pruned_udf, required_features = create_pruned_udf(original_udf, model, predicates)
    parent_op = _.parent

    new_values = {}
    for name, value in parent_op.values.items():
        if name == "prediction":
            udf_kwargs = {
                feat_name: parent_op.values[feat_name]
                for feat_name in required_features
            }
            new_values[name] = pruned_udf(**udf_kwargs)
        else:
            new_values[name] = value

    new_project = ops.Project(parent_op.parent, new_values)

    new_predicates = []
    for pred in _.predicates:
        if isinstance(pred, ops.Less) and isinstance(pred.left, ops.Field):
            new_predicates.append(
                ops.Less(ops.Field(new_project, pred.left.name), pred.right)
            )
        else:
            new_predicates.append(pred)

    return ops.Filter(parent=new_project, predicates=new_predicates)


def optimize_gbdt_expression(expr, model, udf):
    """Optimize an Ibis expression by pruning GBDT models based on filter conditions."""

    op = expr.op()
    new_op = op.replace(prune_gbdt_model, context={"model": model, "original_udf": udf})
    return new_op.to_expr()


def benchmark_sizes(model, predict_gbdt, iterations=3):
    """Benchmark execution times across different Parquet file sizes."""
    file_sizes = [
        # "diamonds_benchmark_1M.parquet",
        "diamonds_benchmark_5M.parquet",
        "diamonds_benchmark_25M.parquet",
        "diamonds_benchmark_100M.parquet",
    ]

    results = {}

    for file_path in file_sizes:
        print(f"\nBenchmarking {file_path}")
        ibis.set_backend("datafusion")
        t = ibis.read_parquet(file_path).drop("prediction", "target")

        result = (
            t.mutate(
                prediction=predict_gbdt(
                    t.carat,
                    t.depth,
                    t.table,
                    t.x,
                    t.y,
                    t.z,
                    t.cut_good,
                    t.cut_ideal,
                    t.cut_premium,
                    t.cut_very_good,
                    t.color_e,
                    t.color_f,
                    t.color_g,
                    t.color_h,
                    t.color_i,
                    t.color_j,
                    t.clarity_if,
                    t.clarity_si1,
                    t.clarity_si2,
                    t.clarity_vs1,
                    t.clarity_vs2,
                    t.clarity_vvs1,
                    t.clarity_vvs2,
                )
            )
            .filter(
                _.carat < 1,
                _.clarity_vvs2 < 1,
                _.color_i < 1,
                _.color_j < 1,
            )
            .select(_.prediction)
        )

        optimized_result = optimize_gbdt_expression(
            result, model=model, udf=predict_gbdt
        )

        regular_times = []
        optimized_times = []

        for i in range(iterations):
            print(f"  Iteration {i + 1}/{iterations}")

            # Regular execution
            start_time = time.time()
            result.execute()
            regular_times.append(time.time() - start_time)

            # Optimized execution
            start_time = time.time()
            optimized_result.execute()
            optimized_times.append(time.time() - start_time)

        results[file_path] = {
            "regular": {
                "mean": sum(regular_times) / len(regular_times),
                "min": min(regular_times),
                "max": max(regular_times),
            },
            "optimized": {
                "mean": sum(optimized_times) / len(optimized_times),
                "min": min(optimized_times),
                "max": max(optimized_times),
            },
        }

        results[file_path]["improvement"] = (
            (
                results[file_path]["regular"]["mean"]
                - results[file_path]["optimized"]["mean"]
            )
            / results[file_path]["regular"]["mean"]
            * 100
        )

    return results


def print_results(results):
    """Print benchmark results in a tabular format."""
    print("\nBenchmark Results:")
    print("-" * 80)
    print(
        f"{'File Size':<20} {'Regular (s)':<15} {'Optimized (s)':<15} {'Improvement':<15}"
    )
    print("-" * 80)

    for file_path, stats in results.items():
        size = file_path.split("_")[2].split(".")[0]
        print(
            f"{size:<20} "
            f"{stats['regular']['mean']:>8.2f} ±{(stats['regular']['max'] - stats['regular']['min']) / 2:>4.2f}   "
            f"{stats['optimized']['mean']:>8.2f} ±{(stats['optimized']['max'] - stats['optimized']['min']) / 2:>4.2f}   "
            f"{stats['improvement']:>8.1f}%"
        )


if __name__ == "__main__":
    input_csv = (
        "data/benches/reg_squarederror/data/diamonds_data_full_trees_100_float64.csv"
    )
    model = PyGradientBoostedDecisionTrees.json_load(
        "data/benches/reg_squarederror/models/diamonds_model_trees_100_float64.json"
    )
    ibis.set_backend("datafusion")
    # uncomment to generate files
    # write_large_parquet(input_csv, 'diamonds_benchmark_1M.parquet', target_rows=1_000_000)
    # write_large_parquet(input_csv, 'diamonds_benchmark_5M.parquet', target_rows=5_000_000)
    # write_large_parquet(input_csv, 'diamonds_benchmark_25M.parquet', target_rows=25_000_000)
    # write_large_parquet(input_csv, 'diamonds_benchmark_100M.parquet', target_rows=100_000_000)

    @ibis.udf.scalar.pyarrow
    def predict_gbdt(
        carat: dt.float64,
        depth: dt.float64,
        table: dt.float64,
        x: dt.float64,
        y: dt.float64,
        z: dt.float64,
        cut_good: dt.float64,
        cut_ideal: dt.float64,
        cut_premium: dt.float64,
        cut_very_good: dt.float64,
        color_e: dt.float64,
        color_f: dt.float64,
        color_g: dt.float64,
        color_h: dt.float64,
        color_i: dt.float64,
        color_j: dt.float64,
        clarity_if: dt.float64,
        clarity_si1: dt.float64,
        clarity_si2: dt.float64,
        clarity_vs1: dt.float64,
        clarity_vs2: dt.float64,
        clarity_vvs1: dt.float64,
        clarity_vvs2: dt.float64,
    ) -> dt.float32:
        array_list = [
            carat,
            depth,
            table,
            x,
            y,
            z,
            cut_good,
            cut_ideal,
            cut_premium,
            cut_very_good,
            color_e,
            color_f,
            color_g,
            color_h,
            color_i,
            color_j,
            clarity_if,
            clarity_si1,
            clarity_si2,
            clarity_vs1,
            clarity_vs2,
            clarity_vvs1,
            clarity_vvs2,
        ]
        return model.predict_arrays(array_list)

    results = benchmark_sizes(model, predict_gbdt)
    print_results(results)
