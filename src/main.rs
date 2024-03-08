mod dataset;
mod model;

use crate::dataset::dataset::get_dataset;
use crate::model::model::get_model;
use std::path::PathBuf;
use tch::nn::{Module, OptimizerConfig};
use tch::Kind::Float;
use tch::{nn, Device, Reduction, Tensor};

const N_EPOCHS: u8 = 10u8;

fn main() {
    println!("Processing dataset...");
    let dataset = get_dataset(
        PathBuf::from("data/train.csv"),
        PathBuf::from("data/test.csv"),
    );

    println!("Setting up net...");
    let vs = nn::VarStore::new(Device::Cpu);
    let net = get_model(&vs.root(), 10, 1);
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    for epoch in 1..N_EPOCHS {
        // println!("{:?}", dataset.train_images);
        // println!("{:?}", dataset.train_labels);
        let res = net.forward(&dataset.train_images);
        // println!("{:?}", res);

        let loss = res.binary_cross_entropy_with_logits::<Tensor>(
            &dataset.train_labels,
            None,
            None,
            Reduction::None,
        ).mean(Float);

        // 反向传播
        opt.backward_step(&loss);

        // 计算精度
        let val_accuracy = res.accuracy_for_logits(&dataset.train_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} val acc: {:5.2}%",
            epoch,
            &loss
            .double_value(&[]),
            100. * &val_accuracy.double_value(&[]),
        );
    }
}
