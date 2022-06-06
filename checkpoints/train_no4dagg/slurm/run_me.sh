#!/bin/bash

source ~/.local/share/virtualenvs/SeparableFlow-wh-ch6f8/bin/activate

echo ==============================

free -h

echo ==============================
echo "start time:"
date
echo ==============================

#cd ~/git/thesis/SeparableFlowThesis
./checkpoints/train_no4dagg/train.sh


echo ==============================
echo "end time:"
date
echo ==============================
