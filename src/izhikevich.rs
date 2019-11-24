
use zerocopy::FromBytes;
use rand::prelude::*;

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

#[allow(dead_code)]
impl Izhikevich {
    pub fn state(&self) -> (f32, f32) {
        (self.v, self.u)
    }
}

/// Creates a randomized set of neurons in accordance with the example code from Izhikevich (2003)
pub fn randomized_neurons(excitatory: usize, inhibitory: usize) -> Vec<Izhikevich> {
    let mut neurons = Vec::with_capacity(excitatory + inhibitory);
    let mut rng = rand::thread_rng();

    for _ in 0..excitatory {
        let noise: f32 = rng.gen();
        let b = 0.2;
        let v = -65.0;
        let n = Izhikevich {
            decay_rate: 0.02,
            sensitivity: b,
            v_reset: v + (15.0 * noise.powi(2)),
            u_reset: 8.0 - (6.0 * noise.powi(2)),
            v,
            u: b * v,
        };

        neurons.push(n);
    }

    for _ in 0..inhibitory {
        let noise: f32 = rng.gen();
        let b = 0.25 - (0.05 * noise);
        let v = -65.0;
        let n = Izhikevich {
            decay_rate: 0.02 + (0.08 * noise),
            sensitivity: b,
            v_reset: v,
            u_reset: 2.0,
            v,
            u: b * v,
        };

        neurons.push(n);
    }

    neurons
}
