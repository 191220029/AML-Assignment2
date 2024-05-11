mod dataset;

use crate::dataset::dataset::fit_dataset;
use std::path::PathBuf;

fn main() {
    println!("Processing dataset...");
    let train_data_file = PathBuf::from("data/fit_train.csv");
    let test_test_file = PathBuf::from("data/fit_test.csv");

    fit_dataset(
        PathBuf::from("data/train.csv"),
        PathBuf::from("data/test.csv"),
        &train_data_file,
        &test_test_file,
    )
    .unwrap();

    let train_data_file = PathBuf::from("data/fit_train.csv");
    let test_test_file = PathBuf::from("data/fit_Churn_Modelling.csv");

    fit_dataset(
        PathBuf::from("data/train.csv"),
        PathBuf::from("data/Churn_Modelling.csv"),
        &train_data_file,
        &test_test_file,
    )
    .unwrap();
}
