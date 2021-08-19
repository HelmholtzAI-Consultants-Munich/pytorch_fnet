#!/bin/bash

conda create -n fnet python=3.7
conda activate fnet
pip install -r requirements.txt
conda install -c pytorch pytorch=1.8.0 torchvision=0.9.0 cuda91=1.0
./scripts/test_run.sh