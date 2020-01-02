use std::iter::FromIterator;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

use super::izhikevich;

/// Currently this is meant to closely replicate the example Matlab code from the paper
/// to ensure that everything is working how it's supposed to
pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    let mut prev_spikes = Array1::<bool>::default(excitatory + inhibitory);

    for _ in 0..time_steps {
        let input =
            thalamic_input(excitatory, inhibitory) + connection_input(&prev_spikes, &connections);

        let spikes: Array1<bool> = neurons
            .iter_mut()
            .enumerate()
            .map(|(i, n)| {
                let i = input[i];
                n.compute_step(i)
            })
            .collect();

        println!("{:?}", spikes);
        let voltages: Vec<f32> = neurons.par_iter().map(|n| n.v).collect();
        println!("{:?}", voltages);
        prev_spikes = spikes;
    }
}

fn thalamic_input(excitatory: usize, inhibitory: usize) -> Array1<f32> {
    let total = excitatory + inhibitory;
    let mut rng = rand::thread_rng();

    Array::from_iter((0..total).map(|i| {
        let noise: f32 = rng.sample(StandardNormal);
        if i < excitatory {
            5.0 * noise
        } else {
            2.0 * noise
        }
    }))
}

fn connection_input(prev_spikes: &Array1<bool>, connections: &Array2<f32>) -> Array1<f32> {
    // this isn't the ideal way of doing this but ndarray doesn't currently support boolean
    // masking so this is the only way to get this to work.

    let prev_spike_indices: Vec<usize> = prev_spikes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            true => Some(i),
            false => None,
        })
        .collect();

    let mut out = Array1::<f32>::zeros(prev_spikes.len());
    for ((y, x), w) in connections.indexed_iter() {
        if prev_spike_indices.contains(&x) {
            out[y] += w
        }
    }

    out
}
