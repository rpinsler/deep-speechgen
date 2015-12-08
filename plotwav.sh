#!/bin/bash

find data/pred/wav16/ -type f -name "*.wav" > split_ids.txt
python fbank_features/signal2logspec.py -p
python fbank_features/logspec_viewer.py
