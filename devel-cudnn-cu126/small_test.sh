#!/bin/bash

# === Works but not perfectly ===


set -euo pipefail

cd /path/to/Megatron-LM
export PBS_O_WORKDIR="$(realpath .)" 
# make sure dataset is in data/

# LD_PRELOAD= \
unset LD_PRELOAD; export LD_PRELOAD=


ml use /soft/modulefiles
ml spack-pe-base/0.8.1
ml use /soft/spack/testing/0.8.1/modulefiles
ml apptainer/main
ml load e2fsprogs

# Avoid OSError: AF_UNIX path too long
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

export BASE_SCRATCH_DIR=/local/scratch/
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR
export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR


# Proxy setup for internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Environment variables for MPI
export ADDITIONAL_PATH=/opt/cray/pe/pals/1.2.12/lib
module load cray-mpich-abi

# For NCCL
module load libfabric

# Set MPI ranks
export NODES=$(wc -l < "$PBS_NODEFILE")
GPUS_PER_NODE=4
export TOTAL_GPUS=$(( NODES * GPUS_PER_NODE ))
export PPN=4
export PROCS=$((NODES * PPN))
echo "NUM_OF_NODES=${NODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, TOTAL_GPUS=${TOTAL_GPUS}, RANKS_PER_NODE=${PPN}, TOTAL_NUM_RANKS=${PROCS}"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # RuntimeError: Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1


# ---- NCCL / OFI envs ----
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} # WARN if you don't want the noise
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=0

# Toggle AWS OFI NCCL plugin ON (1) / OFF (0)
PLUGIN=${PLUGIN:-0}

if [[ "$PLUGIN" == "1" ]]; then
  echo "[INFO] Enabling AWS OFI NCCL Plugin"
    unset NCCL_CROSS_NIC NCCL_COLLNET_ENABLE \
        NCCL_SOCKET_IFNAME NCCL_NSOCKS_PERTHREAD NCCL_SOCKET_NTHREADS

    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}  
    export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws-libfabric-1.22.0/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH # for libfabric
    export LD_LIBRARY_PATH=/eagle/lc-mpi/bharadhwaj/containers/cxi-shim:$LD_LIBRARY_PATH # for libcxi.so.1
    export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

    # echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    # Don't forget to re-export the LD path below inside the container
    export NCCL_NET="AWS Libfabric" # Use OFI if using some other version of libfabric
    export NCCL_OFI_LOG_LEVEL=${NCCL_OFI_LOG_LEVEL:-INFO} # TRACE if having issues
    export FI_PROVIDER=${FI_PROVIDER:-cxi}
    export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export NCCL_COLLNET_ENABLE=0

    # export NCCL_NET_GDR_LEVEL=PHB
    # export NCCL_CROSS_NIC=1
    # export NCCL_COLLNET_ENABLE=1
    # export FI_CXI_DISABLE_HOST_REGISTER=1
    # export FI_MR_CACHE_MONITOR=userfaultfd
    # export FI_CXI_DEFAULT_CQ_SIZE=131072
    # # export NCCL_SOCKET_IFNAME="hsn,bond0"
    # export NCCL_SOCKET_IFNAME=hsn,ib0,ib1
    # export NCCL_NSOCKS_PERTHREAD=4
    # export NCCL_SOCKET_NTHREADS=2
else
  echo "[INFO] Plugin disabled (fallback)."
  unset NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET
  unset FI_PROVIDER FI_CXI_DISABLE_HOST_REGISTER FI_MR_CACHE_MONITOR FI_CXI_DEFAULT_CQ_SIZE
fi

PROFILE=${PROFILE:-0}
if [[ "$PROFILE" == "1" ]]; then
    echo "[INFO] Profiling enabled"
    export NSYS_MPI_STORE_TEAMS_PER_RANK=1
    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH
else
    echo "[INFO] Profiling not enabled"
fi

export NVTE_FUSED_ATTN=0


# ---- Config ----
export RDZV_HOST=$(head -n1 "$PBS_NODEFILE")
export RDZV_PORT=${RDZV_PORT:-29400}
CONTAINER="/path/to/devel-cudnn.sif"
BIND="-B /opt -B /var/run/palsd -B /soft -B /etc/alternatives -B /lus -B /eagle"


echo "[INFO] Using container: $CONTAINER"
echo "[INFO] Env preview:"
echo "  NCCL_DEBUG=$NCCL_DEBUG  PLUGIN=$PLUGIN  PPN=$PPN   "


# ---- Preprocess Data ---- (don't have to do it every time)
# -- MAX seq_len is 1024. DO NOT EXCEED IT bec gpt2 tokenizer
# -- set --max-position-embeddings the same
#   # 1) Fetch TinyStories with HF datasets and write JSONL
#   # 2) Megatron-LM preprocessing
# PREPROCESS_DATA=${PREPROCESS_DATA:-0}
# # Align all these  to the leftmost indent. Do not tab anything
# if [[ "$PREPROCESS_DATA" == "1" ]]; then
# echo "Preprocessing Data..."
# apptainer exec --fakeroot --nv $BIND "$CONTAINER" bash -lc "
# set -euo pipefail
# cd /path/to/Megatron-LM
# mkdir -p data
# python3 - <<'PY'
# from datasets import load_dataset
# ds = load_dataset('roneneldan/TinyStories', split='train')
# if 'text' not in ds.column_names:
#     if 'content' in ds.column_names:
#         ds = ds.rename_column('content', 'text')
#     else:
#         raise SystemExit(f\"Expected 'text' or 'content' in columns: {ds.column_names}\")
# out = 'data/corpus.jsonl'
# ds.to_json(out, lines=True, orient='records', force_ascii=False)
# print(f'Wrote {out} with {len(ds)} rows')
# PY
# python3 tools/preprocess_data.py \
#   --input data/corpus.jsonl \
#   --tokenizer-type HuggingFaceTokenizer \
#   --tokenizer-model gpt2 \
#   --output-prefix data/corpus_tokenized \
#   --append-eod \
#   --workers 4
# "
# else


# Launch: MPICH places processes; torch uses NCCL for comms

# Tried:
# mpiexec nsys apptainer torchrun script - fails at gpu-metrics=all
# mpiexec apptainer nsys torchrun script - appears to work. There is a runtime error but that happens after the .qdrep is generated
#    check report # 6585909 and the 2 before it
# mpiexec apptainer torchrun nsys script - works. There is a rendezvous error when more than 2 nodes are run but everything looks perfect


mpiexec -hostfile "$PBS_NODEFILE" -n "$NODES" -ppn 1 \
  apptainer exec --fakeroot --nv $BIND "$CONTAINER" bash -lc '
set -euo pipefail

# env inside the container
export LD_PRELOAD=
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws-libfabric-1.22.0/lib:/opt/cray/libfabric/1.22.0/lib64:/eagle/lc-mpi/bharadhwaj/containers/cxi-shim:/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
# export NCCL_NET="AWS Libfabric"
# export FI_PROVIDER=${FI_PROVIDER:-cxi}
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export NCCL_COLLNET_ENABLE=0
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH

source /usr/local/venv/bin/activate

torchrun \
  --nproc_per_node='"$PPN"' \
  --nnodes='"$NODES"' \
  --rdzv_backend=c10d \
  --rdzv_endpoint='"$RDZV_HOST:$RDZV_PORT"' \
  py_nsys.py pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --train-iters 1000 \
    --bf16 \
    --optimizer adam \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --lr 6e-4 --min-lr 6e-5 --lr-decay-style cosine \
    --weight-decay 0.1 \
    --init-method-std 0.006 \
    --log-interval 1 \
    --eval-interval 10000 \
    --eval-iters 1 \
    --save-interval 1000000 \
    --data-path data/corpus_tokenized_text_document \
    --split 949,50,1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --num-workers 1 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion
'


echo "[MPIEXEC] JOB ENDED"

