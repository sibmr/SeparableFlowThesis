#!/bin/bash

source ~/.local/share/virtualenvs/SeparableFlow-wh-ch6f8/bin/activate

echo ==============================

free -h

echo ==============================
echo "start time:"
date
pwd
echo "executing:"
echo "./checkpoints/train_originalLongerChairs/train.sh"
echo ==============================

./checkpoints/train_originalLongerChairs/train.sh


echo ==============================
echo "end time:"
date
echo ==============================
