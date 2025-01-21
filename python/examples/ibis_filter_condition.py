from __future__ import annotations

from typing import List

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import _
from ibis.common.patterns import pattern, replace
from ibis.util import Namespace
from trustpy import Feature, PyGradientBoostedDecisionTrees

p = Namespace(pattern, module=ops)

def collect_predicates(filter_op: ops.Filter) -> List[dict]:
    """extract predicates from a Filter operation."""
    predicates = []
    
    for pred in filter_op.predicates:
        if isinstance(pred, ops.Less):
            if isinstance(pred.left, ops.Field):
                predicates.append({
                    'column': pred.left.name,
                    'op': 'Less',
                    'value': pred.right.value if isinstance(pred.right, ops.Literal) else None
                })
    
    return predicates

def create_pruned_udf(original_udf, model, predicates):
    """Create a new UDF using the pruned model based on predicates."""
    print(f"Creating pruned UDF for predicates: {predicates}")
    
    pruned_model = model.prune([
        Feature(pred['column']) < pred['value']
        for pred in predicates
        if pred['op'] == 'Less' and pred['value'] is not None
    ])
    
    required_features = sorted(pruned_model.required_features)
    print(f"Required features after pruning: {required_features}")
    
    feature_names = [model.feature_names[i] for i in required_features]
    print(f"Required feature names: {feature_names}")
    
    param_list = [f"{name}: dt.float64" for name in feature_names]
    params_str = ", ".join(param_list)
    
    code = f"""
def pruned_predict({params_str}) -> dt.float32:
    array_list = [
        {", ".join(feature_names)}
    ]
    predictions = pruned_model.predict_arrays(array_list)
    return predictions
    """
    namespace = {'dt': dt, 'pruned_model': pruned_model}
    exec(code, namespace)
    pruned_predict = namespace['pruned_predict']
    
    pruned_predict = ibis.udf.scalar.pyarrow(pruned_predict)
    
    return pruned_predict, feature_names

@replace(p.Filter)
def prune_gbdt_model(_, **kwargs):
    """
    Rewrite rule to prune GBDT model based on filter predicates.
    """
    model = kwargs['model']
    original_udf = kwargs['original_udf']
    
    predicates = collect_predicates(_)
    print(f"Found predicates: {predicates}")
    
    if not predicates:
        print("No predicates found, returning original filter")
        return _
        
    pruned_udf, required_features = create_pruned_udf(original_udf, model, predicates)
    print(f"Created pruned UDF with features: {required_features}")
    
    parent_op = _.parent
    print(f"Parent operation type: {type(parent_op)}")
    
    new_values = {}
    for name, value in parent_op.values.items():
        if name == 'prediction':
            # Create kwargs for UDF call using required features
            udf_kwargs = {
                feat_name: parent_op.values[feat_name]
                for feat_name in required_features
            }
            new_values[name] = pruned_udf(**udf_kwargs)
            print("Replaced UDF application with pruned version")
        else:
            new_values[name] = value
    
    print("Creating new project operation")
    new_project = ops.Project(parent_op.parent, new_values)
    
    new_predicates = []
    for pred in _.predicates:
        if isinstance(pred, ops.Less):
            if isinstance(pred.left, ops.Field):
                new_predicates.append(
                    ops.Less(
                        ops.Field(new_project, pred.left.name), 
                        pred.right
                    )
                )
            else:
                new_predicates.append(pred)
        else:
            new_predicates.append(pred)
    
    print("Creating new filter operation")
    return ops.Filter(
        parent=new_project,
        predicates=new_predicates
    )

def optimize_gbdt_expression(expr, model, udf):
    """
    Optimize an Ibis expression by pruning GBDT models based on filter conditions.
    """
    print("\nStarting optimization")
    print(f"Original expression: {expr}")
    
    op = expr.op()
    print(f"Operation node type: {type(op)}")
    
    new_op = op.replace(
        prune_gbdt_model,
        context={"model": model, "original_udf": udf}
    )
    
    print(f"New operation node type: {type(new_op)}")
    
    result = new_op.to_expr()
    return result


model = PyGradientBoostedDecisionTrees.json_load(
            "data/benches/reg_squarederror/models/diamonds_model_trees_100_float64.json"
            )

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
    clarity_vvs2: dt.float64
) -> dt.float32:
    array_list = [
            carat, depth, table, x, y, z,
            cut_good, cut_ideal, cut_premium, cut_very_good,
            color_e, color_f, color_g, color_h, color_i, color_j,
            clarity_if, clarity_si1, clarity_si2, clarity_vs1, clarity_vs2,
            clarity_vvs1, clarity_vvs2
        ]
    predictions = model.predict_arrays(array_list)
    return predictions


if __name__ == "__main__":
    ibis.set_backend("datafusion")
    t = ibis.read_csv("data/benches/reg_squarederror/data/diamonds_data_full_trees_100_float64.csv").mutate(depth=_.depth.cast("float64"))
    
    result = t.mutate(
        prediction=predict_gbdt(
            t.carat, t.depth, t.table, t.x, t.y, t.z,
            t.cut_good, t.cut_ideal, t.cut_premium, t.cut_very_good,
            t.color_e, t.color_f, t.color_g, t.color_h, t.color_i, t.color_j,
            t.clarity_if, t.clarity_si1, t.clarity_si2, t.clarity_vs1, t.clarity_vs2,
            t.clarity_vvs1, t.clarity_vvs2
        )
    ).filter( _.clarity_vvs2<1, _.color_i<1, _.color_j<1).select(_.prediction)
    
    optimized_result = optimize_gbdt_expression(
        result,
        model=model,
        udf=predict_gbdt 
    )
    
    optimized_result.execute()
