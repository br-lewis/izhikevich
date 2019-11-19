
use zerocopy::FromBytes;

#[derive(Debug, Copy, Clone, FromBytes)]
#[repr(C)]
pub struct Izhikevich {
    decay_rate: f32,
    sensitivity: f32,

    // mV
    v_reset: f32,
    u_reset: f32,

    // mV
    v: f32,
    u: f32,
}

impl Izhikevich {
    pub fn state(&self) -> (f32, f32) {
        (self.v, self.u)
    }
}

pub fn some_neurons() -> Vec<Izhikevich> {
    vec![
        Izhikevich {
            decay_rate: 0.02,
            sensitivity: 2.0,
            v_reset: -65.0,
            u_reset: 2.0,
            v: -60.0,
            u: -60.0 * 0.2,
        },
        Izhikevich {
            decay_rate: 0.02,
            sensitivity: 0.2,
            v_reset: -65.0,
            u_reset: 8.0,
            v: -60.0,
            u: -60.0 * 0.2,
        },
        Izhikevich {
            decay_rate: 0.02,
            sensitivity: 0.2,
            v_reset: -65.0,
            u_reset: 2.0,
            v: -60.0,
            u: -60.0 * 0.2,
        },
    ]
}
