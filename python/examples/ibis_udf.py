import ibis
import ibis.expr.datatypes as dt
from trustpy import PyGradientBoostedDecisionTrees

ibis.set_backend("datafusion")

model = PyGradientBoostedDecisionTrees.json_load(
            "data/benches/reg_squarederror/models/diamonds_model_trees_100_float64.json"
            )

@ibis.udf.scalar.pyarrow
def predict_gbdt(
    carat: dt.float64,
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
    clarity_vvs2: dt.float64
) -> dt.float32:
    array_list = [
        carat, table, x, y, z,
        cut_good, cut_ideal, cut_premium, cut_very_good,
        color_e, color_f, color_g, color_h, color_i, color_j,
        clarity_if, clarity_si1, clarity_si2, clarity_vs1, clarity_vs2,
        clarity_vvs1, clarity_vvs2
    ]
    predictions = model.predict_arrays(array_list)
    return predictions

t = ibis.read_csv("data/benches/reg_squarederror/data/diamonds_data_full_trees_100_float64.csv")

result = t.mutate(
    prediction=predict_gbdt(
        t.carat, t.table, t.x, t.y, t.z,
        t.cut_good, t.cut_ideal, t.cut_premium, t.cut_very_good,
        t.color_e, t.color_f, t.color_g, t.color_h, t.color_i, t.color_j,
        t.clarity_if, t.clarity_si1, t.clarity_si2, t.clarity_vs1, t.clarity_vs2,
        t.clarity_vvs1, t.clarity_vvs2
    )
)

result.execute()
