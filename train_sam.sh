CUDA_LAUNCH_BLOCKING=1;PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 29666 \
--nproc_per_node=8 \
--use_env u-sam.py \
--num_workers 4 \
--epochs 100 \
--batch_size 24 \
--dataset rectum