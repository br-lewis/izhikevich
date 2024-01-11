use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use structopt::StructOpt;
use tokio::sync::mpsc;

mod cpu;
mod gpu;
mod izhikevich;
mod ui;

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "izhikevich")]
struct Args {
    /// how many timesteps to hold in the buffer (each step is equivalent to 1ms)
    #[structopt(default_value = "1000")]
    steps: usize,

    #[structopt(long = "cpu")]
    use_cpu: bool,

    /// number of excitatory neurons to create
    #[structopt(long = "ne", default_value = "800")]
    num_excitatory: usize,

    /// number of inhibitory neurons to create
    #[structopt(long = "ni", default_value = "200")]
    num_inhibitory: usize,

    /// drawing the spike data live is incredibly slow, set this true to only see
    /// the voltage of neuron 0 over time
    #[structopt(long = "no-spikes", aliases = &["no-spike"])]
    no_spikes: bool,
}

fn main() {
    env_logger::init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .build()
        .unwrap();

    let args: Args = Args::from_args();
    log::info!("{:?}", args);

    let step_buffer_size = args.steps;
    let total_neurons = args.num_excitatory + args.num_inhibitory;

    let (voltage_tx, mut voltage_rx): (mpsc::Sender<f32>, mpsc::Receiver<f32>) = mpsc::channel(1);

    let voltages = Arc::new(Mutex::new(VecDeque::with_capacity(step_buffer_size)));
    let voltage_pusher = Arc::clone(&voltages);

    let (spikes_tx, mut spikes_rx): (mpsc::Sender<Vec<bool>>, mpsc::Receiver<Vec<bool>>) =
        mpsc::channel(1);

    let spikes = Arc::new(Mutex::new(VecDeque::with_capacity(step_buffer_size)));
    let spike_pusher = Arc::clone(&spikes);
    runtime.spawn(async move {
        // doing them both simultaneously keeps the spiking and voltage data matched up
        while let (Some(v), Some(s)) = (voltage_rx.recv().await, spikes_rx.recv().await) {
            let mut voltage_guard = voltage_pusher.lock().unwrap();
            let mut spike_guard = spike_pusher.lock().unwrap();

            if voltage_guard.len() < step_buffer_size {
                voltage_guard.push_back(v)
            } else {
                voltage_guard.pop_front();
                voltage_guard.push_back(v);
            }
            let spike_indices = s
                .iter()
                .enumerate()
                .filter(|(_i, &s)| s)
                .map(|(i, _s)| i as i32)
                .collect();
            if spike_guard.len() < step_buffer_size {
                spike_guard.push_back(spike_indices);
            } else {
                spike_guard.pop_front();
                spike_guard.push_back(spike_indices);
            }
        }
    });

    if args.use_cpu {
        let runner_args = args.clone();
        runtime.spawn(async move {
            let args = runner_args;
            cpu::main(
                args.steps,
                args.num_excitatory,
                args.num_inhibitory,
                voltage_tx,
                spikes_tx,
            )
            .await;
        });
    } else {
        let runner_args = args.clone();
        thread::spawn(move || {
            let args = runner_args;
            runtime.block_on(gpu::main(
                args.steps,
                args.num_excitatory,
                args.num_inhibitory,
                voltage_tx,
                spikes_tx,
            ));
        });
    }

    ui::draw(
        step_buffer_size,
        total_neurons,
        args.no_spikes,
        voltages,
        spikes,
    );
}
