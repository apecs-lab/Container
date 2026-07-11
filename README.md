## Megatron / NCCL Training containers for ALCF nodes

### Container status
- `megatron-cu128`
    - CUDA 12.8
    - torch / transformer-engine - compatible version natively contains in root docker mirror
    - megatron-core 0.11.0

- `devel-cudnn-cu126`
    - CUDA 12.6
    - torch 2.13.0 - not compatible with it's CUDA version, needs to change torch version 
    - Need to install megatron-core in container
