 python -m train.train_mdm --save_dir save/abscontrol_0.5textdrop --dataset humanml --num_steps 400000 --batch_size 64 --resume_checkpoint ./save/model000475000.pt --lr 1e-5 --overwrite --save_interval 10000

python -m train.train_mdm --save_dir save/abscontrol_cubeV2 --dataset humanml --num_steps 400000 --batch_size 64 --resume_checkpoint ./save/model000475000.pt --lr 1e-5 --overwrite --save_interval 10000

 python -m train.train_mdm --save_dir /workspace/writeable/save/abscontrol_cubeV3 --dataset humanml --num_steps 1200000 --batch_size 64 --resume_checkpoint /workspace/writeable/save/model000475000.pt --lr 1e-5 --overwrite --save_interval 10000
