#!/bin/bash

find data/pred/spectrogram/ -type f -name "*.logspec.npy" | tr '.' '\n' | grep "data" > split_ids.txt
python fbank_features/logspec_viewer.py
