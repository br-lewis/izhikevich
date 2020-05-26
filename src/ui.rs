use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;

pub(crate) fn draw(
    time_buffer_size: usize,
    neuron_count: usize,
    no_spikes: bool,
    voltages: Arc<Mutex<VecDeque<f32>>>,
    spikes: Arc<Mutex<VecDeque<Vec<i32>>>>,
) {
    let mut window: PistonWindow = WindowSettings::new("Izhikevich simulation", [1000, 1200])
        .samples(4)
        .build()
        .unwrap();

    window.set_max_fps(60);

    let voltage_reader: Arc<Mutex<VecDeque<f32>>> = Arc::clone(&voltages);
    while let Some(_event) = draw_piston_window(&mut window, |backend| {
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        let (upper, lower) = root.split_vertically(800);

        let mut spike_chart = ChartBuilder::on(&upper)
            .caption("Spikes", ("sans-serif", 10))
            .build_ranged(0..time_buffer_size as i32, 0..neuron_count as i32)?;

        if !no_spikes {
            let spike_guard = spikes.lock().unwrap();

            for (time, spikes) in spike_guard.iter().enumerate() {
                spike_chart.draw_series(PointSeries::of_element(
                    spikes.iter().map(|s| (time as i32, *s)),
                    2,
                    &RED,
                    &|c, s, t| {
                        return EmptyElement::at(c) + Circle::new((0, 0), s, t.filled());
                    },
                ))?;
            }
        }
        spike_chart.configure_mesh().draw()?;

        let mut neuron_chart = ChartBuilder::on(&lower)
            .caption("Neuron 0 voltage", ("sans-serif", 10))
            .build_ranged(0..time_buffer_size as i32, -100f32..30f32)?;

        neuron_chart.configure_mesh().draw()?;
        neuron_chart.draw_series(LineSeries::new(
            voltage_reader
                .lock()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, v)| (i as i32, *v)),
            &RED,
        ))?;

        Ok(())
    }) {
        //println!("{:?}", event);
    }
}
