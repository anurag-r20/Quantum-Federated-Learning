#!/bin/bash
# conda create -n fed python=3.10 && conda activate fed && bash run-gpu.sh
set -e
pip install torch==2.2.2 torchvision==0.17.2
pip install -r requirements.txt
