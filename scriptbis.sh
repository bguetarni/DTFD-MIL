#!/bin/bash
if [ "$1" = "bci" ]; then
    echo "bci dataset detected"
    tmux new-session -d -s 020
elif [ "$1" = "dlbclmorph" ]; then
    echo "dlbclmorph dataset detected"
    tmux new-session -d -s 010
else
    echo "dataset not recognized"
    exit 0
fi

for j in {0..3}
do
    tmux new-window
    tmux send 'cd; source .bashrc; conda activate torch_gpu' ENTER
    if [ "$1" = "bci" ]; then
        g=$((j%2 + 2))
        tmux send "python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --name bci_teacher_$j --gpu $g --dataset bci --stain multi;
        python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --teacher /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis/bci_teacher_$j --name bci_student_$j --gpu $g --dataset bci --stain mono;
        python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/BCI --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --name bci_noKD_$j --gpu $g --dataset bci --stain mono" ENTER
        tmux rename-window bci_$j
    elif [ "$1" = "dlbclmorph" ]; then
        g=$((j%2))
        tmux send "python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --name dlbclmorph_teacher_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain multi; 
        python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --teacher /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KD/dlbclmorph_teacher_$j --name dlbclmorph_student_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain mono;
        python /home/isen/bilel/workspace/DTFD-MIL/custom_main.py --mDATA0_dir_train0 /home/isen/bilel/data/SOA/DTFD-MIL/features/DLBCL-Morph --output /home/isen/bilel/workspace/DTFD-MIL/checkpoints/KDbis --name dlbclmorph_noKD_$j --gpu $g --labels /home/isen/bilel/data/DLBCL-Morph/labels.csv --dataset dlbclmorph --fold fold_$j --stain mono" ENTER
        tmux rename-window dlbclmorph_$j
    fi
done
