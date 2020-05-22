use std::time;

use ndarray::prelude::*;
use ndarray::Zip;
use plotters::prelude::*;
use rayon::prelude::*;
use tokio::sync::mpsc;

use super::izhikevich;
use super::izhikevich::{thalamic_input, Izhikevich};

/// Currently this is meant to closely replicate the example Matlab code from the paper though
/// written in a more object oriented style rather than array oriented to be closer to a
/// theoretically more GPU-friendly style
pub(crate) fn main(
    time_steps: usize,
    excitatory: usize,
    inhibitory: usize,
    voltage_channel: mpsc::Sender<f32>,
    spike_channel: mpsc::Sender<Vec<bool>>,
) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    let mut spikes = Array2::<bool>::default((excitatory + inhibitory, time_steps));
    let mut voltages = Array1::<f32>::zeros(time_steps);

    let mut t: usize = 0;
    //for t in 0..time_steps {
    loop {
        let timer = time::Instant::now();

        let ci = if t == 0 {
            Array1::<f32>::zeros(excitatory + inhibitory)
        } else {
            let prev_column = wrapping_dec(t, time_steps);
            connection_input(&spikes.column(prev_column), &connections)
        };
        let input = thalamic_input(excitatory, inhibitory) + ci;

        let mut new_neurons: Vec<Izhikevich> = Vec::with_capacity(neurons.len());
        let mut current_spikes_buf: Vec<bool> = Vec::with_capacity(neurons.len());

        (0..neurons.len())
            .into_par_iter()
            .zip_eq(0..input.len())
            .map(|(n, i)| {
                let mut neuron = neurons[n];
                let input = input[i];
                let s = neuron.compute_step(input);
                (neuron, s)
            })
            .unzip_into_vecs(&mut new_neurons, &mut current_spikes_buf);

        let current_spikes = Array::from(current_spikes_buf);
        neurons.assign(&Array::from(new_neurons));

        let v = neurons[0].v;
        voltages[t] = v;
        spikes.column_mut(t).assign(&current_spikes);

        let mut vc = voltage_channel.clone();
        tokio::spawn(async move {
            if let Err(_) = vc.send(v).await {
                println!("sending voltage failed");
            }
        });

        let mut sc = spike_channel.clone();
        tokio::spawn(async move {
            if let Err(_) = sc.send(current_spikes.to_vec()).await {
                println!("sending spikes failed");
            }
        });

        t = wrapping_inc(t, time_steps);
        let elapsed = timer.elapsed();
        tokio::spawn(async move {
            println!("{:?}", elapsed);
        });
    }

    //graph_output(graph_file, &spikes, &voltages, &neurons, time_steps);
}

pub fn graph_output(
    graph_file: &str,
    spikes: &Array2<bool>,
    voltages: &Array1<f32>,
    neurons: &Array1<izhikevich::Izhikevich>,
    time_steps: usize,
) {
    let mut spike_points = vec![];
    for t in 0..time_steps {
        let spikes_at = spikes.column(t);
        for i in spike_indices(&spikes_at).iter() {
            spike_points.push((t as f32, *i as f32));
        }
    }

    let root = BitMapBackend::new(graph_file, (1200, 1400)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let (upper, lower) = root.split_vertically(1000);

    let mut spike_chart = ChartBuilder::on(&upper)
        .build_ranged(0f32..time_steps as f32, 0f32..neurons.len() as f32)
        .unwrap();

    spike_chart.configure_mesh().draw().unwrap();
    spike_chart
        .draw_series(PointSeries::of_element(
            spike_points.into_iter(),
            2,
            &RED,
            &|c, s, t| {
                return EmptyElement::at(c) + Circle::new((0, 0), s, t.filled());
            },
        ))
        .unwrap();

    let mut neuron_chart = ChartBuilder::on(&lower)
        .build_ranged(0f32..time_steps as f32, -100f32..30f32)
        .unwrap();

    neuron_chart.configure_mesh().draw().unwrap();
    neuron_chart
        .draw_series(LineSeries::new(
            voltages
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i as f32, *v)),
            &RED,
        ))
        .unwrap();
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

fn wrapping_inc(t: usize, max: usize) -> usize {
    if t == max - 1 {
        0
    } else {
        t + 1
    }
}

fn wrapping_dec(t: usize, max: usize) -> usize {
    if t == 1 {
        max - 1
    } else {
        t - 1
    }
}
