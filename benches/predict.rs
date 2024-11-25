pub mod common;
use common::feature_tree;
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_tree_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_tree_predict");
    for nodes in [15, 31, 63, 127, 255] {
        let feature_count = 10;
        let (tree, features) = feature_tree::create_tree_and_features(nodes, feature_count, 0, 0.0);
        group.bench_function(format!("predict/nodes_{}", nodes), |b| {
            b.iter(|| tree.predict(&features))
        });
    }
    for nan_prob in [0.0, 0.1, 0.25, 0.5] {
        let nodes = 31;
        let feature_count = 10;
        let (tree, features) =
            feature_tree::create_tree_and_features(nodes, feature_count, 0, nan_prob);
        group.bench_function(format!("predict/nan_prob_{}", nan_prob), |b| {
            b.iter(|| tree.predict(&features))
        });
    }
    group.finish();
}

fn benchmark_gbdt_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbdt_predict_arrays");

    for num_trees in [1, 10, 50, 100, 1000, 2000] {
        let nodes_per_tree = 31;
        let feature_count = 10;
        let num_rows = 1000;

        let gbdt = feature_tree::create_gbdt(num_trees, nodes_per_tree, feature_count);
        let feature_arrays = feature_tree::create_feature_arrays(num_rows, feature_count, 0.0);

        group.bench_function(format!("trees_{}", num_trees), |b| {
            b.iter(|| gbdt.predict_arrays(&feature_arrays).unwrap());
        });
    }

    for num_rows in [100, 1000, 10000] {
        let num_trees = 50;
        let nodes_per_tree = 31;
        let feature_count = 10;

        let gbdt = feature_tree::create_gbdt(num_trees, nodes_per_tree, feature_count);
        let feature_arrays = feature_tree::create_feature_arrays(num_rows, feature_count, 0.0);

        group.bench_function(format!("rows_{}", num_rows), |b| {
            b.iter(|| gbdt.predict_arrays(&feature_arrays).unwrap());
        });
    }

    for nan_prob in [0.0, 0.25] {
        let num_trees = 50;
        let nodes_per_tree = 31;
        let feature_count = 10;
        let num_rows = 1000;

        let gbdt = feature_tree::create_gbdt(num_trees, nodes_per_tree, feature_count);
        let feature_arrays = feature_tree::create_feature_arrays(num_rows, feature_count, nan_prob);

        group.bench_function(format!("nan_prob_{}", nan_prob), |b| {
            b.iter(|| gbdt.predict_arrays(&feature_arrays).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = benchmark_tree_predict, benchmark_gbdt_predict
);
criterion_main!(benches);
