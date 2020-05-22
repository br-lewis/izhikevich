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

    /// path to output file that will be created/overwritten
    #[structopt(long = "out", default_value = "out.png")]
    graph_file: String,
}

fn main() {
    env_logger::init();
    let mut runtime = tokio::runtime::Builder::new()
        .threaded_scheduler()
        .build()
        .unwrap();

    let args: Args = Args::from_args();
    log::info!("{:?}", args);

    let time_steps = args.steps;
    let total_neurons = args.num_excitatory + args.num_inhibitory;

    let (voltage_tx, mut voltage_rx): (mpsc::Sender<f32>, mpsc::Receiver<f32>) = mpsc::channel(100);

    let voltages = Arc::new(Mutex::new(VecDeque::with_capacity(time_steps)));
    let voltage_pusher = Arc::clone(&voltages);
    runtime.spawn(async move {
        while let Some(v) = voltage_rx.recv().await {
            let mut guard = voltage_pusher.lock().unwrap();
            if guard.len() < time_steps {
                guard.push_back(v)
            } else {
                guard.pop_front();
                guard.push_back(v);
            }
        }
    });

    let (spikes_tx, mut spikes_rx): (mpsc::Sender<Vec<bool>>, mpsc::Receiver<Vec<bool>>) = mpsc::channel(100);

    let spikes = Arc::new(Mutex::new(VecDeque::with_capacity(time_steps)));
    let spike_pusher = Arc::clone(&spikes);
    runtime.spawn(async move {
        while let Some(s) = spikes_rx.recv().await {
            let mut guard = spike_pusher.lock().unwrap();
            if guard.len() < time_steps {
                guard.push_back(s);
            } else {
                guard.pop_front();
                guard.push_back(s);
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
            );
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

    ui::draw(time_steps, total_neurons, voltages, spikes);
}
