use std::time;

use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use tokio::sync::mpsc;

use super::izhikevich;
use super::izhikevich::{thalamic_input, Izhikevich};

/// Currently this is meant to closely replicate the example Matlab code from the paper though
/// written in a more object oriented style rather than array oriented to be closer to a
/// theoretically more GPU-friendly style
pub(crate) async fn main(
    time_buffer_size: usize,
    excitatory: usize,
    inhibitory: usize,
    voltage_channel: mpsc::Sender<f32>,
    spike_channel: mpsc::Sender<Vec<bool>>,
) {
    let mut neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);

    let mut spikes = Array2::<bool>::default((excitatory + inhibitory, time_buffer_size));
    let mut voltages = Array1::<f32>::zeros(time_buffer_size);

    let mut t: usize = 0;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1));
    loop {
        interval.tick().await;

        let timer = time::Instant::now();

        let ci = if t == 0 {
            Array1::<f32>::zeros(excitatory + inhibitory)
        } else {
            let prev_column = wrapping_dec(t, time_buffer_size);
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

        t = wrapping_inc(t, time_buffer_size);
        let elapsed = timer.elapsed();
        tokio::spawn(async move {
            println!("{:?}", elapsed);
        });
    }

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
