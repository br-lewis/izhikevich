use gnuplot::AxesCommon;
use gnuplot::Figure;
use gnuplot::Fix;
use gnuplot::PlotOption;
use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;

use super::izhikevich;
use super::izhikevich::{thalamic_input, Izhikevich};

/// Currently this is meant to closely replicate the example Matlab code from the paper though
/// written in a more object oriented style rather than array oriented to be closer to a
/// theoretically more GPU-friendly style
pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize, graph_file: &str) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    let mut spikes = Array2::<bool>::default((excitatory + inhibitory, time_steps));
    let mut voltages = Array1::<f32>::zeros(time_steps);

    for t in 0..time_steps {
        let ci = if t == 0 {
            Array1::<f32>::zeros(excitatory + inhibitory)
        } else {
            connection_input(&spikes.column(t - 1), &connections)
        };
        let input = thalamic_input(excitatory, inhibitory) + ci;

        let mut new_neurons: Vec<Izhikevich> = Vec::with_capacity(neurons.len());
        let mut current_spikes: Vec<bool> = Vec::with_capacity(neurons.len());

        (0..neurons.len())
            .into_par_iter()
            .zip_eq(0..input.len())
            .map(|(n, i)| {
                let mut neuron = neurons[n];
                let input = input[i];
                let s = neuron.compute_step(input);
                (neuron, s)
            })
            .unzip_into_vecs(&mut new_neurons, &mut current_spikes);

        let current_spikes = Array::from(current_spikes);
        neurons.assign(&Array::from(new_neurons));

        voltages[t] = neurons[0].v;
        spikes.column_mut(t).assign(&current_spikes);
    }

    graph_output(graph_file, &spikes, &voltages, &neurons, time_steps);
}

pub fn graph_output(
    graph_file: &str,
    spikes: &Array2<bool>,
    voltages: &Array1<f32>,
    neurons: &Array1<izhikevich::Izhikevich>,
    time_steps: usize,
) {
    let mut spike_times = vec![];
    let mut spike_points = vec![];
    for t in 0..time_steps {
        let spikes_at = spikes.column(t);
        for i in spike_indices(&spikes_at).iter() {
            spike_times.push(t);
            // copy out usize to prevent temp value dropped error
            spike_points.push(*i);
        }
    }

    let mut fig = Figure::new();
    fig.axes2d()
        .set_pos(0.0, 0.2)
        .set_size(1.0, 0.8)
        .set_x_range(Fix(0.0), Fix(time_steps as f64))
        .set_y_range(Fix(0.0), Fix(neurons.len() as f64))
        .points(
            &spike_times,
            &spike_points,
            &[PlotOption::PointSymbol('O'), PlotOption::PointSize(0.6)],
        );
    fig.axes2d()
        .set_pos(0.0, 0.0)
        .set_size(1.0, 0.2)
        .set_x_range(Fix(0.0), Fix(time_steps as f64))
        .set_y_range(Fix(-100.0), Fix(30.0))
        .lines(0..time_steps, voltages.iter(), &[]);
    fig.save_to_png(graph_file, 1200, 1400)
        .expect("error writing graph to file");
}

fn connection_input(prev_spikes: &ArrayView1<bool>, connections: &Array2<f32>) -> Array1<f32> {
    // this isn't the ideal way of doing this but ndarray doesn't currently support boolean
    // masking so this is the only way to get this to work.

    let mut out = Vec::with_capacity(prev_spikes.len());

    connections
        .axis_iter(Axis(0)) // iterate across rows
        .into_par_iter()
        .map(|row| {
            // this method is a tiny bit faster than the below but nearly equal
            Zip::from(row)
                .and(prev_spikes)
                .fold(0.0, |acc, w, s| match s {
                    true => acc + w,
                    false => acc,
                })
            /*
            row.iter()
                .zip(prev_spikes)
                .fold(0.0, |acc, (w, s)| match s {
                    true => acc + w,
                    false => acc,
                })
                */
        })
        .collect_into_vec(&mut out);

    Array1::from(out)
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
