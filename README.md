
Implementation of [Izhikevich spiking
neurons][Izhi-2003] running on the CPU
and in compute shaders on the GPU using
[`wgpu-rs`](https://github.com/gfx-rs/wgpu-rs). Mostly an excuse for trying
out compute shaders and GPU programming.

## Milestones ##

### 0.2 ###
Replicate the setup from 0.1 but make it run and graph continuously.

### 0.1 ###
A replication of the example code in the [the paper][Izhi-2003] that produces
a similar graph output.

## Building ##

Should Just Work™ but I found on Windows, compiling `shaderc` from source
used up computer-breaking amounts of RAM so I had to use the
[prebuilt lib](https://github.com/google/shaderc#downloads) and the
[`SHADERC_LIB_DIR` envvar option](https://github.com/google/shaderc-rs#setup).

## Running ##

Default is running on the GPU
```
cargo run -- 1000
```

To use the CPU:
```
cargo run -- --cpu 1000
```

The resulting graph defaults to `./out.png` but can be changed.

Increasing the number of neurons increases RAM usage exponentially due to the
dense connection matrix.

[Izhi-2003]: https://www.izhikevich.org/publications/spikes.pdf