#!/bin/bash
set -e

# Verify PBS environment
if [ -z "$PBS_NODEFILE" ] || [ ! -f "$PBS_NODEFILE" ]; then
    echo "Error: PBS_NODEFILE not found."
    exit 1
fi

# Auto-detect cluster topology
NODES=($(cat "$PBS_NODEFILE" | sort -u))
NUM_NODES=${#NODES[@]}
GPUS_PER_NODE=4
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Scale global batch size automatically based on node count
GLOBAL_BATCH_SIZE=$((16 * NUM_NODES))

# Master node IP setup for HPE Slingshot HSN
FIRST_NODE=$(head -n 1 "$PBS_NODEFILE")
if [[ "$FIRST_NODE" == *"-hsn"* || "$FIRST_NODE" == *".hsn."* ]]; then
    MASTER_ADDR="$FIRST_NODE"
else
    MASTER_ADDR="${FIRST_NODE}-hsn"
fi
MASTER_PORT=29500

CONTAINER_IMAGE="$HOME/container/megatron-cu128.sif"
WORKSPACE_DIR="$HOME/workspace"

echo "=================================================="
echo "Job Metadata Verification:"
echo "  - Total Nodes      : ${NUM_NODES}"
echo "  - Master IP (hsn0) : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  - Global World Size: ${WORLD_SIZE} GPUs"
echo "  - Auto Global Batch: ${GLOBAL_BATCH_SIZE}"
echo "=================================================="

cd "${WORKSPACE_DIR}/Megatron-LM"

# Set Apptainer environment variables
export APPTAINERENV_CUDA_DEVICE_MAX_CONNECTIONS=1
export APPTAINERENV_NCCL_MIN_NCHANNELS=8
export APPTAINERENV_NCCL_NET_GDR_LEVEL=5
export APPTAINERENV_PYTHONPATH="/workspace/Megatron-LM:$PYTHONPATH"

# Launch multi-node training
mpiexec -n ${NUM_NODES} --ppn 1 \
  apptainer exec --nv \
    --bind "${WORKSPACE_DIR}:/workspace" \
    "${CONTAINER_IMAGE}" \
    sh -c "torchrun \
      --nproc_per_node=${GPUS_PER_NODE} \
      --nnodes=${NUM_NODES} \
      --node_rank=\${PALS_NODEID:-\$PMI_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      pretrain_gpt.py \
        --use-mcore-models \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --num-layers 12 \
        --hidden-size 1536 \
        --num-attention-heads 12 \
        --use-flash-attn \
        --transformer-impl transformer_engine \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --train-iters 250 \
        --log-interval 20 \
        --eval-interval 125 \
        --eval-iters 10 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --bf16 \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257"