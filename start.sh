python3 -W ignore /apdcephfs/private_zaixizhang/CoGeneration/main.py
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1 #cuda11必备
#CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m torch.distributed.launch --nproc_per_node  4  --nnodes 1  /apdcephfs/private_zaixizhang/CoGeneration/main.py
