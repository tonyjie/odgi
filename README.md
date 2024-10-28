# odgi-layout-gpu

Computational Pangenoemics is an emerging field that studies genetic variation using a graph structure encom- passing multiple genomes. Visualizing pangenome graphs is vital for understanding genome diversity. Yet, handling large graphs can be challenging due to the high computational demands of the graph layout process. 

Evaluated on 24 human whole-chromosome pangenomes, our GPU-based solution achieves a **57.3x** speedup over the state-of-the-art multithreaded CPU baseline [odgi-layout](https://github.com/pangenome/odgi/blob/master/src/subcommand/layout_main.cpp) without layout quality loss, reducing execution time from hours to minutes.

We build upon the widely-used pangenome tools [ODGI](https://github.com/pangenome/odgi). This codebase includes our GPU kernel implementation of `odgi-layout`. 


## Installation
For CPU-only build, you can follow the installation guide in [ODGI](https://github.com/pangenome/odgi?tab=readme-ov-file#installation). 

Our requirements are the same: 

`odgi` requires a C++ version of 9.3 or higher.

`odgi` pulls in a host of source repositories as dependencies. It may be necessary to install several system-level libraries to build `odgi`. On `Ubuntu 20.04`, these can be installed using `apt`:
```
sudo apt install build-essential cmake python3-distutils python3-dev libjemalloc-dev
```

To build with GPU, we provide a flag `USE_GPU` when `cmake`. 
```
git clone --recursive git@github.com:tonyjie/odgi.git
cd odgi
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Generic -DUSE_GPU=ON
make -j
```

The runnable `odgi` binary should be at `odgi/bin/odgi`

## Usage
To compute `odgi layout` with GPU is very simple: you only need an extra `--gpu`. The other arguments follows the description [here](https://pangenome.github.io/odgi.github.io/rst/commands/odgi_layout.html). 


An example is as follows: 
```
odgi/bin/odgi layout -i ${INPUT_OG} -o ${OUTPUT_LAY} --threads ${NUM_THREAD} --gpu
```


## Upstream PR & Artifact
Ready to be merged into the ODGI with this [PR](https://github.com/pangenome/odgi/pull/593). 

To reproduce the experiments in the paper, the easiet way is to follow our provided artifact [repo](https://github.com/tonyjie/gpu_pangenome_layout_artifact).  


## Paper
**Rapid GPU-Based Pangenome Graph Layout**: https://arxiv.org/abs/2409.00876

To be presented at [SC'24](https://sc24.supercomputing.org/). 