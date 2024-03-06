mod dataset;
mod model;

use tch::{nn, Device};

fn main() {
    println!("Setting up hardware env...");
    let vs = nn::VarStore::new(Device::Cuda(4));
}
