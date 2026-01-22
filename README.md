# Infinite Data, Zero I/O: A CUDA-Optimized Framework for On-the-Fly Synthetic Data Generation for Deep Learning Pretraining (PDP 2026 Accepted Paper)

This repository contains the source code accompanying the paper: "Formula-Driven Supervised Learning at Scale: Overcoming HPC I/O Bottlenecks with Real-Time CUDA Data Synthesis" (submitted to PDP 2026).

The work explores optimizing the real-time, on-the-fly generation of procedural datasets—specifically an extended Formula-Driven Supervised Learning (FDSL) dataset (VisualAtoms)—to bypass traditional I/O and storage bottlenecks in large-scale neural network pretraining on HPC systems.

## 🛠️ Code Structure and Organization

The codebase is organized into four main directories, reflecting the stages of development and analysis: CPU implementation, CUDA optimization, performance profiling, and model integration.

### 💻 CPU Folder (Baseline and NumPy Optimization)

| File/Class | Description | Key Components |
| ---------- | ----------- | -------------- |
| `datagen.py` | Core Python script for generating VisualAtoms | Defines two primary classes: `OriSyntDatasetClass`: Implements the original, unoptimized CPU generation logic and `OptimSyntDatasetClass` Implements our optimized CPU approach using NumPy for efficient vector operations.</li></ul> |
| `profiling.py` | Script used for generating the CPU performance data presented in the paper | Executes generation the built-in profiling option (time.time) available in the `gen_image` function of both classes | 

### ⚡ CUDA Folder (GPU Implementations and Optimization Studies)

This directory contains the various CUDA implementations and kernel optimizations explored during the research, including those selected for the final paper and those investigated as unsuccessful avenues.

General File Structure within Subfolders:

- `vatom.cu` Code for a single-image VisualAtom generation used for early testing and isolated performance analysis.

- `profile.cu` Code dedicated to profiling the specific optimization implemented in the subfolder.

- `exec.sh` Compilation and execution script for the .cu files.

| Folder | Paper Abbreviation | Description |
| ------ | ------------------ | ----------- |
`baseline` | GBASE (GPU Baseline) | The initial, unoptimized CUDA implementation. Includes `singlekernel.cu` for a consolidated kernel approach, simplifying later PyTorch integration |
`memaccess/half2` | MEM | Focuses on optimizing memory access patterns, specifically using the `__half2` data type |
`rng/philox` | RNG | Implements the Philox algorithm for RNG without state reusing |
`batched/philox_rdc` | BAT | Our final, most complex optimization. Includes `multiclass.cu` which implements batched generation with varying parameters per image, crucial for diverse batch creation in training.

### 📈 graphs Folder (Analysis and Visualization)

Contains all the necessary data and scripts to reproduce the figures and analysis presented in the conference paper. `exec.sh` is the main script that runs all necessary matplotlib scripts to generate every graph in the paper.

### 🧠 pretrain Folder (PyTorch Integration and Pretraining)

This directory contains the final deep learning workflow, showing how the optimized data generation pipeline integrates into a standard PyTorch pretraining loop.
General File Structure within Subfolders:

- `pretrain.py`: The main training script. It handles model setup, the training loop, and integration of the custom data pipeline.
- `factory folder`: Contains all the required code for seamless integration into the PyTorch framework (e.g., custom CUDA extensions, `Dataset`/`DataLoader` wrappers).
- `exec.sh`: Provides detailed command-line instructions and examples for calling pretrain.py with various parameters (e.g., specifying optimization level, batch size, etc.).

We have a separate `config` Folder that includes default configurations and parameter sets (e.g., lower compute profiles with reduced max vertices/orbits) for the VisualAtom generation, used for fast testing.

## ⚙️ Usage and Reproducibility: Setting Up the Environment

To replicate the results, run the performance analysis, or train the models, follow these essential steps. This project relies on specific versions of Python and CUDA for guaranteed reproducibility.

### 1. 🐍 Prerequisites and Dependencies

Ensure your system meets the base software requirements before installing the project dependencies.

| Requirement | Version | Purpose |
| ----------- | ------- | ------- |
Python |3.11.7 | The core runtime for high-level logic, analysis, and PyTorch integration |
CUDA Toolkit | 12.1 | Essential for compiling and running the highly optimized C++/CUDA kernels |

### 2. 📦 Installation Steps
Use Git and Pip to set up the environment and download dependencies.

1. Clone the Repository using standard git.
2. `pip install -r requirements.txt`
3. Substitute your virtual environment path on every `exec.sh` script.

### 3. ▶️ Execution Guide

The project is structured to use shell scripts (`exec.sh`) to simplify the compilation and execution of specific components.

1. Navigate to the desired folder (e.g., `cpu`, `cuda/batched`, `philox_rdc`, or `pretrain`).
2. Refer to the local `exec.sh` script for the exact commands used to compile, run, or profile the code in that specific context.