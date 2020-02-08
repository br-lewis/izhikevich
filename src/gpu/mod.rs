use ndarray::prelude::*;

use super::izhikevich;
use super::izhikevich::Izhikevich;

mod gpu_wrapper;

use gpu_wrapper::GpuWrapper;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Config {
    neurons: u32,
    time_step: u32,
}

pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize) {
    let neurons = izhikevich::randomized_neurons(excitatory, inhibitory);
    let connections = izhikevich::randomized_connections(excitatory, inhibitory);
    let spikes = Array2::<u32>::zeros((neurons.len(), time_steps));

    let mut gw = GpuWrapper::new();

    let neuron_buffer = gw.create_buffer(neurons.as_slice().unwrap());
    // this will be created with more permissions than it needs since it's readonly right now
    let connections_buffer = gw.create_buffer(connections.as_slice().unwrap());
    let spike_buffer = gw.create_buffer(&spikes.as_slice().unwrap());

    let config = Config {
        neurons: (excitatory + inhibitory) as u32,
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
                resource: neuron_buffer.binding_resource(),
            },
            wgpu::Binding {
                binding: 2,
                resource: spike_buffer.binding_resource(),
            },
            wgpu::Binding {
                binding: 3,
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
    gw.queue().submit(&[encoder.finish()]);

    for t in 0..time_steps {
        let mut encoder = gw
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        let config = Config {
            neurons: neurons.len() as u32,
            time_step: t as u32,
        };
        let staging_buffer = gw
            .device()
            .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&[config]);

        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &config_storage_buffer,
            0,
            config_buffer_size,
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

    neuron_buffer.staging.map_read_async(
        0,
        neuron_buffer.size,
        |result: wgpu::BufferMapAsyncResult<&[Izhikevich]>| {
            if let Ok(mapping) = result {
                for neuron in mapping.data {
                    println!("{:?}", neuron.state());
                }
            }
        },
    );

    spike_buffer.staging.map_read_async(
        0,
        spike_buffer.size,
        |result: wgpu::BufferMapAsyncResult<&[u32]>| {
            if let Ok(_mapping) = result {
                //println!("Spikes: {:?}", mapping.data);
            }
        },
    );

    // documentation on what exactly this does is sparse but it
    // seems to block until the maps have been read meaning we
    // can read from them multiple times safely
    gw.device().poll(true);
}
