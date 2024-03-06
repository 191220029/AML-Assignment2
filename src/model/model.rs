use tch::nn;
use tch::nn::{LinearConfig, Module};

fn get_model(vs: &nn::Path, in_dim: i64, out_dim: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            in_dim,
            out_dim,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, out_dim, out_dim, Default::default()))
}
