python3 main.py --data_dir $1 \
                --writer_path ./runs \
                --writer_overwrite \
                --model_dir ./checkpoints \
                --n_epochs 100 \
                --warmup_epochs 5 \
                --warmup_lr 1e-9 \
                --opt adan \
                --lr 0.0003 \
                --weight_decay 0.02 \
                --grad_clip 5.0 \
                --batch_size 128 \
                --n_cpu 8
                # --resume ./checkpoints/resnet18/model_last_epoch2_acc47.500.pth