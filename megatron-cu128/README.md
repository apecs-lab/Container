## Getting the Apptainer Image up and running
1. Log into an interactive node (1 node, 25 mins is fine)

1. [Load](https://docs.alcf.anl.gov/polaris/containers/containers/#apptainer-setup) apptainer and the necessary modules 

1. Build the image

    `apptainer build --fakeroot megatron-cu128.sif megatron-cu128.def`

1. Download [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repo with compatible version - tag `core_r0.10.0` for this specific container

    `git clone -b core_r0.10.0 https://github.com/NVIDIA/Megatron-LM.git`

1. Enter the Container `apptainer shell --nv megatron-cu128.sif`

1. (Needed for 1st time enter the container) Export python search path

    `export PYTHONPATH="<path/to/Megatron-LM>:$PYTHONPATH"`

1. Run your training
