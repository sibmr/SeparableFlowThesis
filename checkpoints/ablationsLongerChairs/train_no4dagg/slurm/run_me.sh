#!/bin/bash

source ~/.local/share/virtualenvs/SeparableFlow-wh-ch6f8/bin/activate

echo ==============================

free -h

echo ==============================
echo "start time:"
date
pwd
echo "executing:"
echo "./checkpoints/ablationsLongerChairs/train_no4dagg/train.sh"
echo ==============================

./checkpoints/ablationsLongerChairs/train_no4dagg/train.sh


echo ==============================
echo "end time:"
date
echo ==============================
