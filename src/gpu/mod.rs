use flame as f;
use flamer::flame;

use super::izhikevich;
use super::izhikevich::Izhikevich;

mod gpu_wrapper;

use gpu_wrapper::GpuWrapper;

#[flame]
pub(crate) fn main(time_steps: usize, excitatory: usize, inhibitory: usize) {
    let neurons = izhikevich::randomized_neurons(excitatory, inhibitory);

    let spikes: Vec<u32> = neurons.iter().map(|_| 0).collect();

    let mut gw = GpuWrapper::new();

    let neuron_buffer = gw.create_buffer(neurons.as_slice().unwrap());
    let spike_buffer = gw.create_buffer(&spikes);

    let bind_group_layout =
        gw.device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let bind_group = gw.device().create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: neuron_buffer.binding_resource(),
            },
            wgpu::Binding {
                binding: 1,
                resource: spike_buffer.binding_resource(),
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

    f::start("initial data load");
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
    gw.queue().submit(&[encoder.finish()]);
    f::end("initial data load");

    f::start("time step calculations");
    for _ in 0..time_steps {
        let _g = f::start_guard("time step");

        f::start("enqueue calculation");
        let mut encoder = gw
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(neurons.len() as u32, 1, 1);
        }

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

        gw.queue().submit(&[encoder.finish()]);

        f::end("enqueue calculation");

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
        f::span_of("polling device", || gw.device().poll(true));

    }
    f::end("time step calculations");
}
