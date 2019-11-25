
use rayon::prelude::*;

use super::izhikevich::Izhikevich;

pub(crate) fn main(mut neurons: Vec<Izhikevich>, time_steps: usize) {

    for _ in 0..time_steps {
        let _spikes = neurons.par_iter_mut().map(|n| {
            n.compute_step()
        });
    }
}

