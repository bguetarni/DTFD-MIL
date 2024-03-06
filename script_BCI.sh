#!/bin/bash
tmux new-session -d -s 020
for j in {0..3}
do
    tmux new-window
    tmux send 'cd; source .bashrc; conda activate torch_gpu' ENTER
    g=$((j%2 + 2))
    tmux send "python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --name bci_teacher_$j --gpu $g --dataset bci --stain multi; 
    python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --teacher /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD/bci_teacher_$j --name bci_student_$j --gpu $g --dataset bci --stain mono;
    python /home/isen/bilel/workspace/DTFD-MIL/train.py --data /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD --name bci_noKD_$j --gpu $g --dataset bci --stain mono" ENTER
    tmux rename-window bci_$j
done
