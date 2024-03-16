mod dataset;
mod model;

use crate::dataset::dataset::fit_dataset;
use crate::model::model::get_model;
use std::env::current_dir;
use std::path::PathBuf;
use std::process::Command;
use tch::nn::{Module, OptimizerConfig};
use tch::Kind::Float;
use tch::{nn, Device, Reduction, Tensor};

const N_EPOCHS: u8 = 10u8;

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

    let mut m = Command::new("python");
    m.arg("model.py").arg(train_data_file).arg(test_test_file);

    m.spawn().unwrap().wait().unwrap();

    // println!("Setting up net...");
    // let vs = nn::VarStore::new(Device::Cpu);
    // let net = get_model(&vs.root(), 10, 1);
    // let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    // for epoch in 1..N_EPOCHS {
    //     // println!("{:?}", dataset.train_images);
    //     // println!("{:?}", dataset.train_labels);
    //     let res = net.forward(&dataset.train_images);
    //     // println!("{:?}", res);

    //     let loss = res
    //         .binary_cross_entropy_with_logits::<Tensor>(
    //             &dataset.train_labels,
    //             None,
    //             None,
    //             Reduction::None,
    //         )
    //         .mean(Float);

    //     // loss.backward();

    //     // 反向传播
    //     opt.backward_step(&loss);

    //     // 计算精度
    //     let val_accuracy = res.accuracy_for_logits(&dataset.train_labels);
    //     println!(
    //         "epoch: {:4} train loss: {:8.5} val acc: {:5.2}%",
    //         epoch,
    //         &loss.double_value(&[]),
    //         100. * &val_accuracy.double_value(&[]),
    //     );
    // }
}
