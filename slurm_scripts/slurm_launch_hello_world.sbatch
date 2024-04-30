#!/bin/bash
#SBATCH --job-name=trl_hello_world
#SBATCH --output=/lustre/alignment/alignment_checkpoints/outputs_dey/log_%j.out
#SBATCH --error=/lustre/alignment/alignment_checkpoints/outputs_dey/log_%j.err
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --partition=hippocrates
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --qos=high


JOBNAME=trl_hello_world
DATAPATH=/lustre/alignment/alignment_checkpoints/dey_sources/custom_texts
OUTPUTDIR=/lustre/alignment/alignment_checkpoints/outputs_dey
MODELPATH=/lustre/alignment/alignment_checkpoints/outputs_dey
CODEPATH=/lustre/alignment/alignment_checkpoints/dey_sources/trl/examples
RANK_ZERO_HOST=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
RANK_ZERO_IP=${RANK_ZERO_HOST/ip-/}
RANK_ZERO_IP=${RANK_ZERO_IP//-/.}

export RANK_ZERO_IP_ADDR=$RANK_ZERO_IP

DATALOADER_NUM_WORKERS=0

dist_args="--use_env \
--master_addr=$RANK_ZERO_IP_ADDR \
--nproc_per_node=1 \
--nnodes=\$SLURM_NNODES \
--node_rank=\$SLURM_NODEID"

export_args="--export=ALL,\
PYTHONPATH=$CODEPATH,\
OMP_NUM_THREADS=1,\
NCCL_SHM_DISABLE=1,\
FI_EFA_USE_DEVICE_RDMA=1,\
OPAL_PREFIX=/opt/hpcx/ompi,\
LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib:$LD_LIBRARY_PATH:/usr/local/mpi/lib,\
NCCL_SOCKET_IFNAME=enp71s0,\
FI_PROVIDER=efa"

srun $export_args --container-image=/admin/enroot/saved/pytorchlightning+pytorch_lightning.sqsh --container-mounts=/admin:/admin,/lustre:/lustre bash -c "ulimit -n 65536 && ulimit -a && cd $CODEPATH && pip install trl && python hello_world.py"
