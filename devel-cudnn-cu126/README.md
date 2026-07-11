## Getting the Apptainer Image up and running
1. Log into an interactive node (1 node, 20 mins is fine)
2. [Load](https://docs.alcf.anl.gov/polaris/containers/containers/#apptainer-setup) apptainer and the necessary modules 
3. Build the image: `apptainer build --fakeroot devel-cudnn.sif devel-cudnn.def` 
4. Run your training