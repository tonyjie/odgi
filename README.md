# odgi-layout-gpu

Computational Pangenoemics is an emerging field that studies genetic variation using a graph structure encom- passing multiple genomes. Visualizing pangenome graphs is vital for understanding genome diversity. Yet, handling large graphs can be challenging due to the high computational demands of the graph layout process. 

Evaluated on 24 human whole-chromosome pangenomes, our GPU-based solution achieves a **57.3x** speedup over the state-of-the-art multithreaded CPU baseline [odgi-layout](https://github.com/pangenome/odgi/blob/master/src/subcommand/layout_main.cpp) without layout quality loss, reducing execution time from hours to minutes.

We build upon the widely-used pangenome tools [ODGI](https://github.com/pangenome/odgi). This codebase includes our GPU kernel implementation of `odgi-layout`. 


## Installation
Update: now the GPU-supported build is also integrated into the [ODGI](https://github.com/pangenome/odgi?tab=readme-ov-file#installation). You can follow the ODGI's installation guide to try GPU-accelerated layout easily! 

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

## Performance

Datasets: [HPRC](https://humanpangenome.org/data/), its [Github](https://github.com/human-pangenomics/HPP_Year1_Assemblies)

Machines: 
- 32-thread CPU baseline: 32-core Intel Xeon Gold 6246R CPU@3.4GHz. 
- A6000: NVIDIA RTX A6000 GPU. 
- A100: NVIDIA A100 GPU. 

Run time and speedup — the run time format is in h:mm:ss. 

| Chromosome | CPU      | A6000   | Speedup (A6000) | A100    | Speedup (A100) |
|------------|----------|---------|----------------|---------|----------------|
| Chr.1      | 2:32:38  | 0:04:59 | 30.6x          | 0:02:42 | 56.5x          |
| Chr.2      | 1:17:03  | 0:03:33 | 21.7x          | 0:01:01 | 75.8x          |
| Chr.3      | 1:28:41  | 0:03:27 | 25.7x          | 0:01:31 | 58.5x          |
| Chr.4      | 1:47:32  | 0:03:40 | 29.3x          | 0:02:06 | 51.2x          |
| Chr.5      | 1:41:09  | 0:03:19 | 30.5x          | 0:01:07 | 90.6x          |
| Chr.6      | 1:13:55  | 0:02:49 | 26.3x          | 0:01:27 | 51.0x          |
| Chr.7      | 1:16:46  | 0:03:00 | 25.6x          | 0:01:34 | 49.0x          |
| Chr.8      | 1:17:27  | 0:02:57 | 26.3x          | 0:01:41 | 46.0x          |
| Chr.9      | 1:16:49  | 0:02:53 | 26.6x          | 0:00:55 | 83.8x          |
| Chr.10     | 0:48:34  | 0:02:22 | 20.6x          | 0:00:44 | 66.2x          |
| Chr.11     | 0:56:25  | 0:02:07 | 26.7x          | 0:00:37 | 91.5x          |
| Chr.12     | 0:44:05  | 0:02:07 | 20.9x          | 0:00:49 | 54.0x          |
| Chr.13     | 1:03:32  | 0:02:22 | 26.8x          | 0:00:53 | 71.9x          |
| Chr.14     | 0:51:21  | 0:02:04 | 24.9x          | 0:00:46 | 67.0x          |
| Chr.15     | 1:11:33  | 0:02:52 | 25.0x          | 0:01:16 | 56.5x          |
| Chr.16     | 2:19:47  | 0:04:56 | 28.3x          | 0:12:58 | 10.8x          |
| Chr.17     | 1:03:45  | 0:02:01 | 31.7x          | 0:01:07 | 57.1x          |
| Chr.18     | 0:50:29  | 0:01:50 | 27.6x          | 0:01:08 | 44.6x          |
| Chr.19     | 0:40:23  | 0:01:29 | 27.3x          | 0:00:27 | 89.8x          |
| Chr.20     | 0:51:34  | 0:01:30 | 34.3x          | 0:01:01 | 50.7x          |
| Chr.21     | 0:44:18  | 0:01:26 | 30.9x          | 0:00:38 | 69.9x          |
| Chr.22     | 0:39:59  | 0:01:37 | 24.8x          | 0:00:30 | 80.0x          |
| Chr.X      | 1:04:06  | 0:01:49 | 35.4x          | 0:00:49 | 78.4x          |
| Chr.Y      | 0:01:55  | 0:00:03 | 36.9x          | 0:00:04 | 28.7x          |
| **Geometric Mean** |  |         | **27.7x**      |         | **57.3x**      |



## Upstream PR & Artifact
Ready to be merged into the ODGI with this [PR](https://github.com/pangenome/odgi/pull/593). 

To reproduce the experiments in the paper, the easiet way is to follow our provided artifact [repo](https://github.com/tonyjie/gpu_pangenome_layout_artifact).  


## Paper
**Rapid GPU-Based Pangenome Graph Layout**: https://ieeexplore.ieee.org/abstract/document/10793207

Presented at [SC'24](https://sc24.conference-program.com/presentation/?id=pap443&sess=sess382). 

## Cite
```
@inproceedings{li2024rapid,
  title={Rapid GPU-Based Pangenome Graph Layout},
  author={Li, Jiajie and Schmelzle, Jan-Niklas and Du, Yixiao and Heumos, Simon and Guarracino, Andrea and Guidi, Giulia and Prins, Pjotr and Garrison, Erik and Zhang, Zhiru},
  booktitle={SC24: International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--19},
  year={2024},
  organization={IEEE}
}
```
