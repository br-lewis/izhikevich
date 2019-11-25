use clap::arg_enum;
use structopt::StructOpt;

mod cpu;
mod gpu;
mod izhikevich;

use izhikevich::randomized_neurons;

arg_enum! {
    #[derive(Debug)]
    enum ComputationType {
        Cpu,
        Gpu,
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "izhikevich")]
struct Args {
    #[structopt(default_value = "1")]
    steps: usize,

    #[structopt(long = "comp-type",
        possible_values = &ComputationType::variants(),
        case_insensitive = true,
        default_value = "gpu")]
    comp_type: ComputationType,

    /// Number of excitatory neurons to create
    #[structopt(long = "ne",
        default_value = "800")]
    num_excitatory: usize,

    /// Number of inhibitory neurons to create
    #[structopt(long = "ni",
        default_value = "200")]
    num_inhibitory: usize,
    
}

fn main() {
    env_logger::init();

    let args = Args::from_args();

    log::info!("{:?}", args);

    let neurons = randomized_neurons(args.num_excitatory, args.num_inhibitory);

    match args.comp_type {
        ComputationType::Cpu => cpu::main(neurons, args.steps),
        ComputationType::Gpu => gpu::main(neurons, args.steps),
    }
}
