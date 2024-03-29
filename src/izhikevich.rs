use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

#[derive(Debug, Copy, Clone, FromZeroes, FromBytes, AsBytes)]
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

impl Izhikevich {
    pub fn compute_step(&mut self, i: f32) -> bool {
        let spike = if self.v >= 30.0 {
            self.v = self.v_reset;
            self.u = self.u + self.u_reset;
            true
        } else {
            false
        };

        self.v += 0.5 * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i);
        self.v += 0.5 * (0.04 * self.v.powi(2) + 5.0 * self.v + 140.0 - self.u + i);
        self.u += self.decay_rate * (self.sensitivity * self.v - self.u);

        spike
    }
}

/// Creates a randomized set of neurons in accordance with the example code from Izhikevich (2003)
pub fn randomized_neurons(excitatory: usize, inhibitory: usize) -> Array1<Izhikevich> {
    let total = excitatory + inhibitory;
    let mut rng = rand::thread_rng();

    Array::from_iter((0..total).map(|i| {
        let noise: f32 = rng.gen();
        if i <= excitatory {
            let b = 0.2;
            let v = -65.0;
            Izhikevich {
                decay_rate: 0.02,
                sensitivity: b,
                v_reset: v + (15.0 * noise.powi(2)),
                u_reset: 8.0 - (6.0 * noise.powi(2)),
                v,
                u: b * v,
            }
        } else {
            let b = 0.25 - (0.05 * noise);
            let v = -65.0;
            Izhikevich {
                decay_rate: 0.02 + (0.08 * noise),
                sensitivity: b,
                v_reset: v,
                u_reset: 2.0,
                v,
                u: b * v,
            }
        }
    }))
}

pub fn randomized_connections(excitatory: usize, inhibitory: usize) -> Array2<f32> {
    // The Matlab code declares the connection matrix as
    // S=[0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];
    // which results in a matrix of shape(Ne+Ni, Ne+Ni) with the second array
    // appended "to the right" of the first
    // i.e. if
    // Ne=2, Ni=1
    // [ones(Ne+Ni, Ne), zeros(Ne+Ni, Ni)]
    // would be
    //    1   1   0
    //    1   1   0
    //    1   1   0

    let mut rng = rand::thread_rng();

    let total = excitatory + inhibitory;

    let mut connections: Array2<f32> = Array::zeros((total, total));

    for ((_y, x), v) in connections.indexed_iter_mut() {
        if x < excitatory {
            let noise: f32 = rng.gen();
            *v = 0.5 * noise;
        } else {
            let noise: f32 = rng.gen();
            *v = -1.0 * noise;
        }
    }

    connections
}

pub fn thalamic_input(excitatory: usize, inhibitory: usize) -> Array1<f32> {
    let total = excitatory + inhibitory;
    let mut rng = rand::thread_rng();

    Array::from_iter((0..total).map(|i| {
        let noise: f32 = rng.sample(StandardNormal);
        if i < excitatory {
            5.0 * noise
        } else {
            2.0 * noise
        }
    }))
}
