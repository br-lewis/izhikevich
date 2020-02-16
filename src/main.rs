use clap::arg_enum;
use structopt::StructOpt;

mod cpu;
mod gpu;
mod izhikevich;

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
    /// how many timesteps to simulate (each step is equivalent to 1ms)
    #[structopt(default_value = "1")]
    steps: usize,

    /// whether to use the CPU or GPU for neuron timestep computation
    #[structopt(long = "comp-type",
        possible_values = &ComputationType::variants(),
        case_insensitive = true,
        default_value = "cpu")]
    comp_type: ComputationType,

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

    let args = Args::from_args();

    log::info!("{:?}", args);

    match args.comp_type {
        ComputationType::Cpu => cpu::main(
            args.steps,
            args.num_excitatory,
            args.num_inhibitory,
            &args.graph_file,
        ),
        ComputationType::Gpu => gpu::main(
            args.steps,
            args.num_excitatory,
            args.num_inhibitory,
            &args.graph_file,
        ),
    }
}
