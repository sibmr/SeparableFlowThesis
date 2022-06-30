#!/bin/bash

source ~/.local/share/virtualenvs/SeparableFlow-wh-ch6f8/bin/activate

echo ==============================

free -h

echo ==============================
echo "start time:"
date
pwd
echo "executing:"
echo "./checkpoints/debug/train_original4GPU/train.sh"
echo ==============================

./checkpoints/debug/train_original4GPU/train.sh


echo ==============================
echo "end time:"
date
echo ==============================
