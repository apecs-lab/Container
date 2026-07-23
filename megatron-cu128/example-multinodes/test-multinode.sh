#!/bin/bash
set -e

# Verify PBS environment
if [ -z "$PBS_NODEFILE" ] || [ ! -f "$PBS_NODEFILE" ]; then
    echo "Error: PBS_NODEFILE not found."
    exit 1
fi

# Node and cluster topology setup
NODES=($(cat "$PBS_NODEFILE" | sort -u))
NUM_NODES=${#NODES[@]}
GPUS_PER_NODE=4
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Clean Master node hostname parsing for Slingshot HSN
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
echo "=================================================="

cd "${WORKSPACE_DIR}/Megatron-LM"

# Apptainer environment exports
export APPTAINERENV_CUDA_DEVICE_MAX_CONNECTIONS=1
export APPTAINERENV_NCCL_MIN_NCHANNELS=8
export APPTAINERENV_NCCL_NET_GDR_LEVEL=5
export APPTAINERENV_PYTHONPATH="/workspace/Megatron-LM:$PYTHONPATH"

# Launch multi-node training via sh -c to evaluate node_rank on each target node
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
        --num-layers 8 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --use-flash-attn \
        --transformer-impl transformer_engine \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 2 \
        --global-batch-size 16 \
        --train-iters 100 \
        --eval-iters 10 \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --bf16 \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50257"