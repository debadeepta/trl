#!/bin/bash
#SBATCH --job-name=trl_hello_world
#SBATCH --output=/lustre/alignment/alignment_checkpoints/outputs_dey/log_%j.out
#SBATCH --error=/lustre/alignment/alignment_checkpoints/outputs_dey/log_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --partition=hippocrates
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --qos=high


JOBNAME=trl_rm
OUTPUTDIR=/lustre/alignment/alignment_checkpoints/outputs_dey
CODEPATH=/lustre/alignment/alignment_checkpoints/dey_sources/trl
DATAPATH=/lustre/alignment/alignment_checkpoints/datasets/15-march_rep_only_1.5k_neel_4k_shuf_for_RM.hf
BBPATH=/lustre/alignment/alignment_checkpoints/dey_sources/bitsandbytes
RANK_ZERO_HOST=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
RANK_ZERO_IP=${RANK_ZERO_HOST/ip-/}
RANK_ZERO_IP=${RANK_ZERO_IP//-/.}

export RANK_ZERO_IP_ADDR=$RANK_ZERO_IP
export GPUS_PER_NODE=8

#DATALOADER_NUM_WORKERS=0

# dist_args="--use_env \
# --master_addr=$RANK_ZERO_IP_ADDR \
# --nproc_per_node=1 \
# --nnodes=\$SLURM_NNODES \
# --node_rank=\$SLURM_NODEID"

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $RANK_ZERO_IP \
    --main_process_port 29500 \
    "

export SCRIPT="scripts/reward_modeling_hippo.py"
export SCRIPT_ARGS=" \
    --model_name_or_path=facebook/opt-350m \
    --output_dir=$OUTPUT_DIR/"reward_model" \
    --dataset_path=$DATAPATH \
    --per_device_train_batch_size=64 \
    --num_train_epochs=10 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=False \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_steps=100 \
    --evaluation_strategy="steps" \
    --max_length=512 \
    --lora_task_type SEQ_CLS
    "

export_args="--export=ALL,\
WANDB_API_KEY=wandbkey,\
WANDB_PROJECT=dey_project,\
PYTHONPATH=$CODEPATH,\
OMP_NUM_THREADS=1,\
"

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

srun $export_args --container-image=/admin/enroot/saved/ngc_pytorch:24.04.sqsh --container-mounts=/admin:/admin,/lustre:/lustre bash -c "ulimit -n 65536 && ulimit -a && cd $BBPATH && pip install -r requirements-dev.txt && cmake -DCOMPUTE_BACKEND=cuda -S . && make && pip install . && cd $CODEPATH && pip install --upgrade trl && pip install wandb && $CMD"
