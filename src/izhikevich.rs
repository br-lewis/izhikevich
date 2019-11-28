use rand::prelude::*;
use zerocopy::FromBytes;

#[derive(Debug, Copy, Clone, FromBytes)]
#[repr(C)]
pub struct Izhikevich {
    pub decay_rate: f32,
    pub sensitivity: f32,

    // mV
    pub v_reset: f32,
    pub u_reset: f32,

    // mV
    pub v: f32,
    pub u: f32,
}

#[allow(dead_code)]
impl Izhikevich {
    pub fn state(&self) -> (f32, f32) {
        (self.v, self.u)
    }

    pub fn compute_step(&mut self, i: f32) -> bool {
        let mut v_next = 0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i;
        let mut u_next = self.decay_rate * (self.sensitivity * self.v - self.u);

        let spike = if v_next >= 30.0 {
            v_next = self.v_reset;
            u_next = u_next + self.u_reset;
            true
        } else {
            false
        };

        self.v = v_next;
        self.u = u_next;

        spike
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

pub fn randomized_connections(excitatory: usize, inhibitory: usize) -> Vec<Vec<f32>> {
    // The Matlab code declares the connection matrix as
    // S=[0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];
    // which results in a matrix of shape(Ne+Ni, Ne+Ni) with the second array
    // appended "to the right" of the first
    // i.e. if it
    // Ne=2, Ni=1
    // [ones(Ne+Ni, Ne), zeros(Ne+Ni, Ni)]
    // would be
    //    1   1   0
    //    1   1   0
    //    1   1   0

    let mut rng = rand::thread_rng();

    let total = excitatory + inhibitory;
    let mut connectivity: Vec<Vec<f32>> = Vec::with_capacity(total);

    for _ in 0..total {
        let mut row = Vec::with_capacity(total);
        for _ in 0..excitatory {
            let noise: f32 = rng.gen();
            let weight = 0.5 * noise;
            row.push(weight);
        }

        for _ in 0..inhibitory {
            let noise: f32 = rng.gen();
            row.push( -1.0 * noise );
        }

        connectivity.push(row);
    }

    connectivity
}
