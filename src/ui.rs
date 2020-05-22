use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use plotters::prelude::*;
use piston_window::{EventLoop, PistonWindow, WindowSettings};

pub(crate) fn draw(time_steps: usize, neuron_count: usize, voltages: Arc<Mutex<VecDeque<f32>>>) {

    let mut window: PistonWindow = WindowSettings::new("Izhikevich simulation", [1000, 1200])
        .samples(4)
        .build()
        .unwrap();

    window.set_max_fps(60);

    let voltage_reader: Arc<Mutex<VecDeque<f32>>> = Arc::clone(&voltages);
    while let Some(_event) = draw_piston_window(&mut window, |backend| {
        let spike_points: Vec<(i32, i32)> = vec![];

        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        let (upper, lower) = root.split_vertically(800);

        let mut spike_chart = ChartBuilder::on(&upper)
            .caption("Spikes", ("sans-serif", 10))
            .build_ranged(0..time_steps as i32, 0..neuron_count as i32)?;

        spike_chart.configure_mesh().draw()?;
        spike_chart.draw_series(PointSeries::of_element(
            spike_points.into_iter(),
            2,
            &RED,
            &|c, s, t| {
                return EmptyElement::at(c) + Circle::new((0, 0), s, t.filled());
            },
        ))?;

        let mut neuron_chart = ChartBuilder::on(&lower)
            .caption("Neuron 0 voltage", ("sans-serif", 10))
            .build_ranged(0..time_steps as i32, -100f32..30f32)?;

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