use std::iter::FromIterator;

use gnuplot::AxesCommon;
use gnuplot::Figure;
use gnuplot::PlotOption;

use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

use super::izhikevich;

/// Currently this is meant to closely replicate the example Matlab code from the paper
/// to ensure that everything is working how it's supposed to
pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize, graph_file: &str) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    let mut prev_spikes = Array2::<bool>::default((excitatory + inhibitory, time_steps));
    let mut voltages: Array1<f32> = Array1::zeros(time_steps);

    for t in 0..time_steps {
        let input = thalamic_input(excitatory, inhibitory)
            + connection_input(&prev_spikes.column(t), &connections);

        let spikes: Array1<bool> = neurons
            .iter_mut()
            .enumerate()
            .map(|(i, n)| {
                let i = input[i];
                n.compute_step(i)
            })
            .collect();

        voltages[t] = neurons[0].v;
        prev_spikes.column_mut(t).assign(&spikes);
    }

    let mut spike_times = vec![];
    let mut spike_points = vec![];
    for t in 0..time_steps {
        let spikes_at = prev_spikes.column(t);
        for i in spike_indices(&spikes_at).iter() {
            spike_times.push(t);
            // copy out usize to prevent temp value dropped error
            spike_points.push(*i);
        }
    }

    let mut fig = Figure::new();
    fig.axes2d().set_title("asdf", &[]).points(
        spike_times,
        spike_points,
        &[PlotOption::PointSymbol('.')],
    );
    fig.save_to_png(graph_file, 800, 1000)
        .expect("error writing graph to file");
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

fn connection_input(prev_spikes: &ArrayView1<bool>, connections: &Array2<f32>) -> Array1<f32> {
    // this isn't the ideal way of doing this but ndarray doesn't currently support boolean
    // masking so this is the only way to get this to work.

    let prev_spike_indices = spike_indices(prev_spikes).to_vec();

    let mut out = Array1::<f32>::zeros(prev_spikes.len());
    for ((y, x), w) in connections.indexed_iter() {
        if prev_spike_indices.contains(&x) {
            out[y] += w
        }
    }

    out
}

fn spike_indices(arr: &ArrayView1<bool>) -> Array1<usize> {
    arr.iter()
        .enumerate()
        .filter_map(|(i, s)| match s {
            true => Some(i),
            false => None,
        })
        .collect()
}
