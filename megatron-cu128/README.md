## Getting the Apptainer Image up and running
1. Log into an interactive node (1 node, 25 mins is fine)

2. [Load](https://docs.alcf.anl.gov/polaris/containers/containers/#apptainer-setup) apptainer and the necessary modules 

3. Build the image
    ```bash
    apptainer build --fakeroot megatron-cu128.sif megatron-cu128.def
    ```

4. Download [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repo with compatible version - tag `core_r0.11.0` for this specific container
    ```bash
    mkdir -p $HOME/workspace
    cd $HOME/workspace
    git clone -b core_r0.11.0 [https://github.com/NVIDIA/Megatron-LM.git](https://github.com/NVIDIA/Megatron-LM.git)
    ```

5. Enter the Container (No need for torchdynamo error suppression flags anymore, as the Triton prober bug is fixed during build)
    ```bash
    apptainer shell --nv \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1,PYTHONPATH="/workspace:$PYTHONPATH" \
    --bind $HOME/workspace:/workspace \
    megatron-cu128.sif
    ```

6. Run your training
    > Smallest test run (Note: `--vocab-size` is required when using NullTokenizer with mock data)
    ```bash
    cd /workspace/Megatron-LM

    torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 pretrain_gpt.py \
      --use-mcore-models \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --num-layers 4 \
      --hidden-size 512 \
      --num-attention-heads 8 \
      --seq-length 1024 \
      --max-position-embeddings 1024 \
      --micro-batch-size 4 \
      --global-batch-size 16 \
      --train-iters 20 \
      --lr 0.00015 \
      --lr-decay-style cosine \
      --bf16 \
      --mock-data \
      --tokenizer-type NullTokenizer \
      --vocab-size 50257 \
      --no-masked-softmax-fusion \
      --no-bias-gelu-fusion \
      --transformer-impl transformer_engine
    ```

## Notes

### Container Env
- **CUDA_DEVICE_MAX_CONNECTIONS=1**: Required when scaling distributed pretraining with Megatron-Core. It serializes kernel launches to allow asynchronous execution streams (compute GEMMs and NCCL collectives) to overlap safely without causing hardware connection timeouts or deadlocks (cluster hangs).

- **For Pure NCCL Network Profiling (e.g., raw `nccl-tests`):**
  Do not set `CUDA_DEVICE_MAX_CONNECTIONS=1` globally as it restricts the hardware to a single connection channel. Use maximum parallel channels instead to stress-test peak interconnect bandwidth on HPE Slingshot:
  ```bash
  NCCL_MIN_NCHANNELS=8 NCCL_NET_GDR_LEVEL=5 mpirun ...
  ```

- **For Megatron-Core Production Pretraining:**
  Combine the connection guard with maximum logical network channels at the execution line:
  ```bash
  CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_MIN_NCHANNELS=8 torchrun pretrain_gpt.py ...
  ```