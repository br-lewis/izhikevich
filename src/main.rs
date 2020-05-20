use structopt::StructOpt;

mod cpu;
mod gpu;
mod izhikevich;

#[derive(Debug, StructOpt)]
#[structopt(name = "izhikevich")]
struct Args {
    /// how many timesteps to simulate (each step is equivalent to 1ms)
    #[structopt(default_value = "1")]
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

#[tokio::main]
async fn main() {
    env_logger::init();

    let args = Args::from_args();

    log::info!("{:?}", args);

    if args.use_cpu {
        cpu::main(
            args.steps,
            args.num_excitatory,
            args.num_inhibitory,
            &args.graph_file,
        );
    } else {
        gpu::main(
            args.steps,
            args.num_excitatory,
            args.num_inhibitory,
            &args.graph_file,
        ).await;
    }
}
