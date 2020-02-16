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
    #[structopt(default_value = "1")]
    steps: usize,

    #[structopt(long = "comp-type",
        possible_values = &ComputationType::variants(),
        case_insensitive = true,
        default_value = "cpu")]
    comp_type: ComputationType,

    /// Number of excitatory neurons to create
    #[structopt(long = "ne", default_value = "800")]
    num_excitatory: usize,

    /// Number of inhibitory neurons to create
    #[structopt(long = "ni", default_value = "200")]
    num_inhibitory: usize,

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
