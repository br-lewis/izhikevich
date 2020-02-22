use std::iter::FromIterator;

use ndarray::prelude::*;

use super::izhikevich;
use super::izhikevich::Izhikevich;

mod gpu_wrapper;

use gpu_wrapper::GpuWrapper;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Config {
    neurons: u32,
    total_time_steps: u32,
    time_step: u32,
}

pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize, graph_file: &str) {
    let neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);
    let spikes = Array2::<u32>::zeros((neurons.len(), time_steps));

    let mut gw = GpuWrapper::new();

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

    let config_staging_buffer = gw
        .device()
        .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&[config]);

    let config_buffer_size = std::mem::size_of::<Config>() as wgpu::BufferAddress;

    let config_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        size: config_buffer_size,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let initial_thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);
    let thalamic_buffer_size =
        (initial_thalamic_input.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    // thalamic input uses random noise which is hard to do on a GPU so we generate it on the CPU
    // and copy it over every time step
    let thalamic_staging_buffer = gw
        .device()
        .create_buffer_mapped(
            initial_thalamic_input.len(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST,
        )
        .fill_from_slice(&initial_thalamic_input.as_slice().unwrap());

    let thalamic_storage_buffer = gw.device().create_buffer(&wgpu::BufferDescriptor {
        size: thalamic_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout =
        gw.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                    },
                    wgpu::BindGroupLayoutBinding {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutBinding {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutBinding {
                        binding: 3,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    },
                    wgpu::BindGroupLayoutBinding {
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

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

    for t in 0..time_steps {
        let config = Config {
            neurons: neurons.len() as u32,
            total_time_steps: time_steps as u32,
            time_step: t as u32,
        };

        let thalamic_input = izhikevich::thalamic_input(excitatory, inhibitory);

        let mut encoder = gw
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        let config_staging_buffer = gw
            .device()
            .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&[config]);

        let input_buffer = gw
            .device()
            .create_buffer_mapped(thalamic_input.len(), wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&thalamic_input.as_slice().unwrap());

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

        gw.queue().submit(&[encoder.finish()]);
    }
    let mut encoder = gw
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

    encoder.copy_buffer_to_buffer(
        &neuron_buffer.storage,
        0,
        &neuron_buffer.staging,
        0,
        neuron_buffer.size,
    );
    encoder.copy_buffer_to_buffer(
        &spike_buffer.storage,
        0,
        &spike_buffer.staging,
        0,
        spike_buffer.size,
    );

    gw.queue().submit(&[encoder.finish()]);

    neuron_buffer.staging.map_read_async(
        0,
        neuron_buffer.size,
        |result: wgpu::BufferMapAsyncResult<&[Izhikevich]>| {
            if let Ok(_mapping) = result {
                /*
                for neuron in mapping.data {
                    println!("{:?}", neuron.state());
                }
                */
            }
        },
    );

    let graph_file = graph_file.to_owned();
    spike_buffer.staging.map_read_async(
        0,
        spike_buffer.size,
        move |result: wgpu::BufferMapAsyncResult<&[u32]>| {
            if let Ok(mapping) = result {
                let spikes_per_time: Array2<u32> =
                    Array::from_shape_vec((neurons.len(), time_steps), mapping.data.to_vec())
                        .unwrap();
                super::cpu::graph_output(
                    &graph_file,
                    &spikes_per_time.map(|&x| if x > 0 { true } else { false }),
                    &Array::from_iter((0..time_steps).map(|_| 0.0)),
                    &neurons,
                    time_steps,
                );
            }
        },
    );

    // documentation on what exactly this does is sparse but it
    // seems to block until the maps have been read meaning we
    // can read from them multiple times safely
    gw.device().poll(true);
}
