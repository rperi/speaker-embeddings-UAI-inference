#!/bin/bash

KALDI_ROOT=/home/raghu/tools/kaldi/

export PATH=$PATH:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/ivectorbin

export PYTHONPATH=$PWD:$PWD/predict/:$PWD/predict/unified_adversarial_invariance/:$PWD/predict/unified_adversarial_invariance/model_configs:$PWD/predict/unified_adversarial_invariance/datasets/
