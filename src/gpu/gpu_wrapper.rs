use std::mem;

use zerocopy::AsBytes;

pub struct GpuWrapper {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
}

impl GpuWrapper {
    pub fn new() -> Self {
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions::default(),
            wgpu::BackendBit::PRIMARY,
        )
        .expect("error creating adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        });

        let cs_module = device.create_shader_module(&Self::izhikevich_shader());

        GpuWrapper {
            device,
            queue,
            shader: cs_module,
        }
    }

    pub fn create_buffer<T: 'static + Copy + AsBytes>(&self, data: &[T]) -> BufferWrapper {
        let staging_buffer = self.device().create_buffer_with_data(
            data.as_bytes(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        let size = (data.len() * mem::size_of::<T>()) as wgpu::BufferAddress;

        let storage_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            size: size,
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        BufferWrapper {
            staging: staging_buffer,
            storage: storage_buffer,
            size: size,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&mut self) -> &mut wgpu::Queue {
        &mut self.queue
    }

    pub fn shader(&self) -> &wgpu::ShaderModule {
        &self.shader
    }

    fn izhikevich_shader() -> Vec<u32> {
        let cs = include_bytes!(concat!(env!("OUT_DIR"), "/", "izhikevich.comp.spv"));
        wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap()
    }
}

pub struct BufferWrapper {
    pub staging: wgpu::Buffer,
    pub storage: wgpu::Buffer,
    pub size: u64,
}

impl BufferWrapper {
    pub fn binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer {
            buffer: &self.storage,
            range: 0..self.size,
        }
    }
}
