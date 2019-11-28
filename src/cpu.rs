
use rayon::prelude::*;

use super::izhikevich;

//pub(crate) fn main(mut neurons: Vec<Izhikevich>, connections: Vec<Vec<f32>>, time_steps: usize) {
pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    for _ in 0..time_steps {
        let _spikes: Vec<bool> = neurons.par_iter_mut().enumerate().map(|(i, n)| {
            let i = -2.0;
            n.compute_step(i)
        }).collect();
    }
}

