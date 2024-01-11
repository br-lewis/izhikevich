use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use minifb::{Window, WindowOptions};
use plotters::backend::BGRXPixel;
use plotters::prelude::*;
use zerocopy::AsBytes;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

pub(crate) fn draw(
    time_buffer_size: usize,
    neuron_count: usize,
    no_spikes: bool,
    voltages: Arc<Mutex<VecDeque<f32>>>,
    spikes: Arc<Mutex<VecDeque<Vec<i32>>>>,
) {
    let mut img_buf: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let window = Window::new("Izhikevich", WIDTH, HEIGHT, WindowOptions::default())
        .expect("error creating window");

    let voltage_reader: Arc<Mutex<VecDeque<f32>>> = Arc::clone(&voltages);

    while window.is_open() {
        let root = BitMapBackend::<BGRXPixel>::with_buffer_and_format(
            img_buf.as_bytes_mut(),
            (WIDTH as u32, HEIGHT as u32),
        )
        .expect("error creating bitmap backend")
        .into_drawing_area();
        root.fill(&WHITE).expect("error filling bitmap background");

        let (upper, lower) = root.split_vertically(800);

        let mut spike_chart = ChartBuilder::on(&upper)
            .caption("Spikes", ("sans-serif", 10))
            .build_cartesian_2d(0..time_buffer_size as i32, 0..neuron_count as i32)
            .expect("error building chart");

        if !no_spikes {
            let spike_guard = spikes.lock().unwrap();

            for (time, spikes) in spike_guard.iter().enumerate() {
                spike_chart
                    .draw_series(PointSeries::of_element(
                        spikes.iter().map(|s| (time as i32, *s)),
                        2,
                        &RED,
                        &|c, s, t| {
                            return EmptyElement::at(c) + Circle::new((0, 0), s, t.filled());
                        },
                    ))
                    .expect("error drawing spike chart");
            }
        }
        spike_chart
            .configure_mesh()
            .draw()
            .expect("error drawing spike chart mesh");

        let mut neuron_chart = ChartBuilder::on(&lower)
            .caption("Neuron 0 voltage", ("sans-serif", 10))
            .build_cartesian_2d(0..time_buffer_size as i32, -100f32..30f32)
            .expect("error building chart");

        neuron_chart
            .configure_mesh()
            .draw()
            .expect("error drawing voltage chart mesh");
        neuron_chart
            .draw_series(LineSeries::new(
                voltage_reader
                    .lock()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as i32, *v)),
                &RED,
            ))
            .expect("error drawing voltage");
    }
}
