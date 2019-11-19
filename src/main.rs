use std::env;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::path::Path;

mod izhikevich;

use izhikevich::{Izhikevich, some_neurons};

const SHADER_DIR: &str = "shaders";
const SHADER_FILE: &str = "izhikevich.comp";

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: wgpu-test <timesteps>");
        std::process::exit(1);
    }
    let steps = args[1].parse::<usize>().expect("invalid timesteps value");

    let neurons = some_neurons();
    let spikes: Vec<u32> = neurons.iter().map(|_| 0).collect();

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .expect("error creating adapter");

    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    let cs = compile_shader();
    let cs_module = device.create_shader_module(cs.as_binary());

    let neuron_staging_buffer = device
        .create_buffer_mapped(
            neurons.len(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        )
        .fill_from_slice(&neurons);

    let neuron_size = (neurons.len() * mem::size_of::<Izhikevich>()) as wgpu::BufferAddress;

    let neuron_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: neuron_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let spike_staging_buffer = device
        .create_buffer_mapped(
            neurons.len(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        )
        .fill_from_slice(&spikes);

    let spike_size = (spikes.len() * mem::size_of::<u32>()) as wgpu::BufferAddress;

    let spike_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: spike_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            wgpu::BindGroupLayoutBinding {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &neuron_storage_buffer,
                    range: 0..neuron_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &spike_storage_buffer,
                    range: 0..spike_size,
                },
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    for _ in 0..steps {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        encoder.copy_buffer_to_buffer(
            &neuron_staging_buffer,
            0,
            &neuron_storage_buffer,
            0,
            neuron_size,
        );
        encoder.copy_buffer_to_buffer(
            &spike_staging_buffer,
            0,
            &spike_storage_buffer,
            0,
            spike_size,
        );

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(neurons.len() as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &neuron_storage_buffer,
            0,
            &neuron_staging_buffer,
            0,
            neuron_size,
        );
        encoder.copy_buffer_to_buffer(
            &spike_storage_buffer,
            0,
            &spike_staging_buffer,
            0,
            spike_size,
        );

        queue.submit(&[encoder.finish()]);

        neuron_staging_buffer.map_read_async(
            0,
            neuron_size,
            |result: wgpu::BufferMapAsyncResult<&[Izhikevich]>| {
                if let Ok(mapping) = result {
                    //println!("Neurons: {:?}", mapping.data)
                    for neuron in mapping.data {
                        println!("{:?}", neuron.state());
                    }
                }
            },
        );

        spike_staging_buffer.map_read_async(
            0,
            spike_size,
            |result: wgpu::BufferMapAsyncResult<&[u32]>| {
                if let Ok(mapping) = result {
                    println!("Spikes: {:?}", mapping.data);
                }
            },
        );

        // documentation on what exactly this does is sparse but it
        // seems to block until the maps have been read meaning we 
        // can read from them multiple times safely
        device.poll(true);
    }
}

fn compile_shader() -> shaderc::CompilationArtifact {
    let path = Path::new(SHADER_DIR).join(SHADER_FILE);
    let mut f = File::open(path).expect("unable to open shader file");
    let mut buf: String = String::new();
    f.read_to_string(&mut buf)
        .expect("unable to read shader file");

    let mut compiler = shaderc::Compiler::new().expect("error creating shader compiler");
    let options = shaderc::CompileOptions::new().expect("error creating shader compiler options");

    compiler
        .compile_into_spirv(
            &buf,
            shaderc::ShaderKind::Compute,
            SHADER_FILE,
            "main",
            Some(&options),
        )
        .expect("error compiling shader")
}
