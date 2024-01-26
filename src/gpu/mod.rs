use std::convert::TryInto;
use std::num::NonZeroU32;
use std::sync::mpsc::sync_channel;
use std::time;

use ndarray::prelude::*;
use tokio::sync::{mpsc, oneshot};
use wgpu::util::DeviceExt;
use zerocopy::AsBytes;

use super::izhikevich;
//use super::izhikevich::Izhikevich;

mod gpu_wrapper;

use gpu_wrapper::GpuWrapper;

#[derive(Debug, Copy, Clone, AsBytes)]
#[repr(C)]
struct Config {
    neurons: u32,
    total_time_steps: u32,
    time_step: u32,
}

pub(crate) async fn main(
    time_buffer_size: usize,
    excitatory: usize,
    inhibitory: usize,
    voltage_channel: mpsc::Sender<f32>,
    spike_channel: mpsc::Sender<Vec<bool>>,
) {
    let neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);
    let spikes = Array2::<u32>::zeros((time_buffer_size, neurons.len()));

    let mut gw: GpuWrapper = GpuWrapper::new().await;

    let neuron_buffer = gw.create_buffer("neurons", neurons.as_slice().unwrap());
    // this will be created with more permissions than it needs since it's readonly right now
    let connections_buffer = gw.create_buffer("connections", connections.as_slice().unwrap());
    let spike_buffer = gw.create_buffer("spikes", &spikes.as_slice().unwrap());

    // TODO: try to generally clean up the rest of the buffer and bind group creation
    let config = Config {
        neurons: (excitatory + inhibitory) as u32,
        total_time_steps: time_buffer_size as u32,
        time_step: 0,
    };

    /*
       let config_staging_buffer = gw
           .device()
           .create_buffer_init(&wgpu::util::BufferInitDescriptor {
               label: Some("config_staging"),
               contents: config.as_bytes(),
               usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
           });
    */
    let config_buffer_size = std::mem::size_of::<Config>() as wgpu::BufferAddress;

    let config_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("config_storage"),
        size: config_buffer_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let initial_thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);
    let thalamic_buffer_size =
        (initial_thalamic_input.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    // thalamic input uses random noise which is hard to do on a GPU so we generate it on the CPU
    // and copy it over every time step
    /*
       let thalamic_staging_buffer =
           gw.device()
               .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                   label: Some("thalamic_staging"),
                   contents: initial_thalamic_input.as_slice().unwrap().as_bytes(),
                   usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
               });
    */
    let thalamic_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("thalamic_storage"),
        size: thalamic_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout =
        gw.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("izhikevich_sim_step_bind_group_layout"),
                entries: &[
                    // config buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(config_buffer_size),
                        },
                    },
                    // thalamic
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        //count: NonZeroU32::new(initial_thalamic_input.len() as u32),
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    // neuron
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        // count: NonZeroU32::new(neurons.len() as u32),
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    // spike
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        //count: NonZeroU32::new(spikes.len() as u32),
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    // connections
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        // count: NonZeroU32::new(connections.len() as u32),
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            });

    let bind_group = gw.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    config_storage_buffer.as_entire_buffer_binding(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(
                    thalamic_storage_buffer.as_entire_buffer_binding(),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: neuron_buffer.binding_resource(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spike_buffer.binding_resource(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: connections_buffer.binding_resource(),
            },
        ],
    });

    let pipeline_layout = gw
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline"),
            push_constant_ranges: &[],
            bind_group_layouts: &[&bind_group_layout],
        });

    let compute_pipeline = gw
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("izhikevich_timestep"),
            layout: Some(&pipeline_layout),
            module: gw.shader(),
            entry_point: "main",
        });

    let mut voltages: Vec<f32> = Vec::with_capacity(time_buffer_size);

    let mut t: usize = 0;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1));
    loop {
        interval.tick().await;
        let timer = time::Instant::now();

        let config = Config {
            neurons: neurons.len() as u32,
            total_time_steps: time_buffer_size as u32,
            time_step: t as u32,
        };

        let thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);

        let mut encoder = gw
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("time step {}", t)),
            });

        let config_staging_buffer =
            gw.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("config_staging"),
                    contents: config.as_bytes(),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });

        let input_buffer = gw
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("thalamic_input"),
                contents: thalamic_input.as_slice().unwrap().as_bytes(),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        encoder.copy_buffer_to_buffer(
            &config_staging_buffer,
            0,
            &config_storage_buffer,
            0,
            config_buffer_size,
        );

        encoder.copy_buffer_to_buffer(
            &input_buffer,
            0,
            &thalamic_storage_buffer,
            0,
            thalamic_buffer_size,
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute pass descriptor"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(neurons.len() as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &neuron_buffer.storage,
            0,
            &neuron_buffer.staging,
            0,
            std::mem::size_of::<izhikevich::Izhikevich>() as wgpu::BufferAddress,
        );

        encoder.copy_buffer_to_buffer(
            &spike_buffer.storage,
            (t * neurons.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            &spike_buffer.staging,
            (t * neurons.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            (neurons.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
        );

        gw.queue().submit(Some(encoder.finish()));

        {
            let (neuron_tx, mut neuron_rx) = oneshot::channel();
            let (spike_tx, mut spike_rx) = oneshot::channel();
            //read first neuron voltage
            let neuron_time_slice = neuron_buffer.staging.slice(..);
            neuron_time_slice.map_async(wgpu::MapMode::Read, move |result| {
                neuron_tx.send(result).unwrap();
            });
            let spike_time_slice = spike_buffer.staging.slice(..);
            spike_time_slice.map_async(wgpu::MapMode::Read, move |result| {
                spike_tx.send(result).unwrap();
            });

            gw.device().poll(wgpu::Maintain::Wait);

            neuron_rx.try_recv().unwrap().unwrap();
            let data = neuron_time_slice.get_mapped_range();
            let raw: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            let v = raw[4];
            voltages.push(v);

            let vc = voltage_channel.clone();
            if let Err(_) = vc.send(v).await {
                println!("sending voltage failed");
            }

            spike_rx.try_recv().unwrap().unwrap();
            let data = spike_time_slice.get_mapped_range();
            let spikes: Vec<bool> = data
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .map(|v| if v > 0 { true } else { false })
                .collect();

            let sc = spike_channel.clone();
            if let Err(_) = sc.send(spikes).await {
                println!("sending spikes failed");
            }
        }

        neuron_buffer.staging.unmap();
        spike_buffer.staging.unmap();

        t = wrapping_inc(t, time_buffer_size);

        let elapsed = timer.elapsed();
        /*
            tokio::spawn(async move {
                println!("{:?}", elapsed);
            });
        */
    }
}

fn wrapping_inc(t: usize, max: usize) -> usize {
    if t == max - 1 {
        0
    } else {
        t + 1
    }
}
