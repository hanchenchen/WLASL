CUDA_VISIBLE_DEVICES=7 nohup python train_I3D.py > logdir/logs/1223-189-i3d-cls=100-output 2>&1 &  \
# --load-weights "work-dir/dist-gpu=2xbs=1-train/1009-21-wo-SimsiamAlign-18 18.9/dev_0.18900_epoch38_model.pt" & \
# --phase test  & \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
