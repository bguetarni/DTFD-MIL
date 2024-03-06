#!/bin/bash
tmux new-session -d -s 010
for j in {0..3}
do
    tmux new-window
    tmux send 'cd; source .bashrc; conda activate torch_gpu' ENTER
    g=$((j%2))
    tmux send "python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --name dlbclmorph_teacher_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain multi; 
    python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --teacher /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD/dlbclmorph_teacher_$j --name dlbclmorph_student_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain mono;
    python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --name dlbclmorph_noKD_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain mono" ENTER
    tmux rename-window dlbclmorph_$j
done
