use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use minifb::{Window, WindowOptions};
use plotters::backend::BGRXPixel;
use plotters::prelude::*;

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;

pub(crate) fn draw(
    time_buffer_size: usize,
    neuron_count: usize,
    no_spikes: bool,
    voltages: Arc<Mutex<VecDeque<f32>>>,
    spikes: Arc<Mutex<VecDeque<Vec<i32>>>>,
) {
    let mut img_buf = BufferWrapper(vec![0; WIDTH * HEIGHT]);

    let mut window = Window::new("Izhikevich", WIDTH, HEIGHT, WindowOptions::default())
        .expect("error creating window");
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    {
        let root = BitMapBackend::<BGRXPixel>::with_buffer_and_format(
            img_buf.borrow_mut(),
            (WIDTH as u32, HEIGHT as u32),
        )
        .expect("error creating bitmap backend")
        .into_drawing_area();
        root.fill(&WHITE).expect("error filling bitmap background");

        let (upper, lower) = root.split_vertically(800);

        upper.fill(&WHITE).expect("error initializing spike chart");

        lower.fill(&WHITE).expect("error initializing lower chart");

        root.present().expect("error presenting ui");
    }
    let voltage_reader: Arc<Mutex<VecDeque<f32>>> = Arc::clone(&voltages);

    while window.is_open() {
        {
            let root = BitMapBackend::<BGRXPixel>::with_buffer_and_format(
                img_buf.borrow_mut(),
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

            root.present().expect("error presenting ui");
        }

        window
            .update_with_buffer(img_buf.borrow(), WIDTH, HEIGHT)
            .expect("error updating buffer");
    }
}

struct BufferWrapper(Vec<u32>);
impl Borrow<[u8]> for BufferWrapper {
    fn borrow(&self) -> &[u8] {
        // Safe for alignment: align_of(u8) <= align_of(u32)
        // Safe for cast: u32 can be thought of as being transparent over [u8; 4]
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, self.0.len() * 4) }
    }
}
impl BorrowMut<[u8]> for BufferWrapper {
    fn borrow_mut(&mut self) -> &mut [u8] {
        // Safe for alignment: align_of(u8) <= align_of(u32)
        // Safe for cast: u32 can be thought of as being transparent over [u8; 4]
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut u8, self.0.len() * 4) }
    }
}
impl Borrow<[u32]> for BufferWrapper {
    fn borrow(&self) -> &[u32] {
        self.0.as_slice()
    }
}
impl BorrowMut<[u32]> for BufferWrapper {
    fn borrow_mut(&mut self) -> &mut [u32] {
        self.0.as_mut_slice()
    }
}
