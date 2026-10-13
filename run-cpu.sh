#!/bin/bash
# conda create -n fed python=3.10 && conda activate fed && bash run-cpu.sh
set -e
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
