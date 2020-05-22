use std::convert::TryInto;
use std::time;

use ndarray::prelude::*;
use tokio::sync::mpsc;
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
    time_steps: usize,
    excitatory: usize,
    inhibitory: usize,
    voltage_channel: mpsc::Sender<f32>,
    spike_channel: mpsc::Sender<Vec<bool>>,
) {
    let neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);
    let spikes = Array2::<u32>::zeros((neurons.len(), time_steps));

    let mut gw: GpuWrapper = GpuWrapper::new().await;

    let neuron_buffer = gw.create_buffer(neurons.as_slice().unwrap());
    // this will be created with more permissions than it needs since it's readonly right now
    let connections_buffer = gw.create_buffer(connections.as_slice().unwrap());
    let spike_buffer = gw.create_buffer(&spikes.as_slice().unwrap());

    // TODO: try to generally clean up the rest of the buffer and bind group creation
    let config = Config {
        neurons: (excitatory + inhibitory) as u32,
        total_time_steps: time_steps as u32,
        time_step: 0,
    };

    let config_staging_buffer = gw.device().create_buffer_with_data(
        config.as_bytes(),
        wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC,
    );

    let config_buffer_size = std::mem::size_of::<Config>() as wgpu::BufferAddress;

    let config_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("config storage"),
        size: config_buffer_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let initial_thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);
    let thalamic_buffer_size =
        (initial_thalamic_input.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    // thalamic input uses random noise which is hard to do on a GPU so we generate it on the CPU
    // and copy it over every time step
    let thalamic_staging_buffer = gw.device().create_buffer_with_data(
        initial_thalamic_input.as_slice().unwrap().as_bytes(),
        wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST,
    );

    let thalamic_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("thalamic storage"),
        size: thalamic_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout =
        gw.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                bindings: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                ],
            });

    let bind_group = gw.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &config_storage_buffer,
                    range: 0..config_buffer_size,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &thalamic_storage_buffer,
                    range: 0..thalamic_buffer_size,
                },
            },
            wgpu::Binding {
                binding: 2,
                resource: neuron_buffer.binding_resource(),
            },
            wgpu::Binding {
                binding: 3,
                resource: spike_buffer.binding_resource(),
            },
            wgpu::Binding {
                binding: 4,
                resource: connections_buffer.binding_resource(),
            },
        ],
    });

    let pipeline_layout = gw
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

    let compute_pipeline = gw
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: gw.shader(),
                entry_point: "main",
            },
        });

    let mut encoder = gw
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("init data entry"),
        });

    encoder.copy_buffer_to_buffer(
        &neuron_buffer.staging,
        0,
        &neuron_buffer.storage,
        0,
        neuron_buffer.size,
    );
    encoder.copy_buffer_to_buffer(
        &spike_buffer.staging,
        0,
        &spike_buffer.storage,
        0,
        spike_buffer.size,
    );
    encoder.copy_buffer_to_buffer(
        &connections_buffer.staging,
        0,
        &connections_buffer.storage,
        0,
        connections_buffer.size,
    );
    encoder.copy_buffer_to_buffer(
        &config_staging_buffer,
        0,
        &config_storage_buffer,
        0,
        config_buffer_size,
    );
    encoder.copy_buffer_to_buffer(
        &thalamic_staging_buffer,
        0,
        &thalamic_staging_buffer,
        0,
        thalamic_buffer_size,
    );

    gw.queue().submit(&[encoder.finish()]);

    let mut voltages: Vec<f32> = Vec::with_capacity(time_steps);

    let mut t: usize = 0;
    //for t in 0..time_steps {
    loop {
        let timer = time::Instant::now();

        let config = Config {
            neurons: neurons.len() as u32,
            total_time_steps: time_steps as u32,
            time_step: t as u32,
        };

        let thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);

        let mut encoder = gw
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("time step {}", t)),
            });

        let config_staging_buffer = gw
            .device()
            .create_buffer_with_data(config.as_bytes(), wgpu::BufferUsage::COPY_SRC);

        let input_buffer = gw.device().create_buffer_with_data(
            thalamic_input.as_slice().unwrap().as_bytes(),
            wgpu::BufferUsage::COPY_SRC,
        );

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
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(neurons.len() as u32, 1, 1);
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

        gw.queue().submit(&[encoder.finish()]);

        //read first neuron voltage

        let neuron_future = neuron_buffer.staging.map_read(
            0,
            std::mem::size_of::<izhikevich::Izhikevich>() as wgpu::BufferAddress,
        );
        let spike_future = spike_buffer.staging.map_read(
            (t * neurons.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            (neurons.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
        );
        gw.device().poll(wgpu::Maintain::Wait);

        if let Ok(mapping) = neuron_future.await {
            let raw: Vec<f32> = mapping
                .as_slice()
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            //println!("{} {}", raw.len(), raw[4]);
            let v = raw[4];
            voltages.push(v);

            let mut vc = voltage_channel.clone();
            //tokio::spawn(async move {
                if let Err(_) = vc.send(v).await {
                    println!("sending voltage failed");
                }
            //});
        }

        if let Ok(mapping) = spike_future.await {
            let spikes: Vec<bool> = mapping
                .as_slice()
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .map(|v| if v > 0 { true } else { false })
                .collect();

            let mut sc = spike_channel.clone();
            //tokio::spawn(async move {
                if let Err(_) = sc.send(spikes).await {
                    println!("sending spikes failed");
                }
            //});
        }

        t = wrapping_inc(t, time_steps);

        let elapsed = timer.elapsed();
        tokio::spawn(async move {
            println!("{:?}", elapsed);
        });
    }
    /*

    let mut encoder = gw
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("final data extract"),
        });

    encoder.copy_buffer_to_buffer(
        &spike_buffer.storage,
        0,
        &spike_buffer.staging,
        0,
        spike_buffer.size,
    );

    gw.queue().submit(&[encoder.finish()]);

    let graph_file = graph_file.to_owned();

    // this seems to work on Mac just fine but Windows will stall sometimes in release mode
    let spike_future = spike_buffer.staging.map_read(0, spike_buffer.size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    gw.device().poll(wgpu::Maintain::Wait);

    if let Ok(mapping) = spike_future.await {

        let raw: Vec<u32> = mapping
            .as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let spikes_per_time: Array2<u32> =
            Array::from_shape_vec((neurons.len(), time_steps), raw).unwrap();

        super::cpu::graph_output(
            &graph_file,
            &spikes_per_time.map(|&x| if x > 0 { true } else { false }),
            &Array::from_iter(voltages.into_iter()),
            &neurons,
            time_steps,
        );
    }
    */
}

fn wrapping_inc(t: usize, max: usize) -> usize {
    if t == max - 1 {
        0
    } else {
        t + 1
    }
}
