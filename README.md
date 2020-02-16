
Implementation of [Izhikevich spiking
neurons](https://www.izhikevich.org/publications/spikes.pdf) running on the CPU
and in compute shaders on the GPU using
[`wgpu-rs`](https://github.com/gfx-rs/wgpu-rs).

Right now this is just a replication of the example code in the paper and is
mostly an excuse for trying out compute shaders and GPU programming.

## Dependencies ##

This requires `gnuplot` be available in the `PATH` when plotting output.

## Building ##

Should Just Work™ but I found on Windows, compiling `shaderc` from source
used up computer-breaking amounts of RAM so I had to use the
[prebuilt lib](https://github.com/google/shaderc#downloads) and the
[`SHADERC_LIB_DIR` envvar option](https://github.com/google/shaderc-rs#setup).

## Running ##

Default is running on the CPU
```
cargo run -- 1000
```

To use a GPU:
```
cargo run -- --comp-type gpu 1000
```

The resulting graph defaults to `./out.png` but can be changed. The output
when running on the GPU does not have voltage data from the first neuron unlike
the CPU because currently it doesn't track historical neuron state.

Increasing the number of neurons increases RAM usage exponentially due to the
dense connection matrix.
