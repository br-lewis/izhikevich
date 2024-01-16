use std::mem;

use wgpu::util::DeviceExt;
use zerocopy::AsBytes;

pub struct GpuWrapper {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
}

impl GpuWrapper {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::DEBUG,
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("error creating adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device descriptor"),
                    features: wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("");

        let cs_module = device.create_shader_module(Self::izhikevich_shader());

        GpuWrapper {
            device,
            queue,
            shader: cs_module,
        }
    }

    pub fn create_buffer<T: 'static + Copy + AsBytes>(
        &self,
        name: &str,
        data: &[T],
    ) -> BufferWrapper {
        let size = (data.len() * mem::size_of::<T>()) as wgpu::BufferAddress;

        let staging_buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_staging", name)),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_buffer = self
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{}_storage", name)),
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        BufferWrapper {
            staging: staging_buffer,
            storage: storage_buffer,
            size,
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

    fn izhikevich_shader() -> wgpu::ShaderModuleDescriptor<'static> {
        wgpu::include_spirv!(concat!(env!("OUT_DIR"), "/", "izhikevich.comp.spv"))
    }
}

pub struct BufferWrapper {
    pub staging: wgpu::Buffer,
    pub storage: wgpu::Buffer,
    pub size: u64,
}

impl BufferWrapper {
    pub fn binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer(self.storage.as_entire_buffer_binding())
    }
}
