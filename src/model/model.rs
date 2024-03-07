use tch::nn;
use tch::nn::Module;

const HIDDEN_NODES: i64 = 1024;

pub fn get_model(vs: &nn::Path, in_dim: i64, out_dim: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            in_dim,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(
            vs / "layer2",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(
            vs / "layer3",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(
            vs / "layer4",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(
            vs / "layer5",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(vs, HIDDEN_NODES, out_dim, Default::default()))
}
