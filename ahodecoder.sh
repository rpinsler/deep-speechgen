#!/bin/sh

basedir="data/pred"
nlines=${1:--0}
files=$(ls "$basedir"/lf0/*.lf0 | head -n $nlines);

for f in $files; do
	fileid=$(echo $f | tr '/' '\n' | tail -1 | tr '.' '\n' | head -1)
	ahocoder_64/ahodecoder16_64 "$basedir/lf0/$fileid.lf0" "$basedir/mcp/$fileid.mcp" "$basedir/mfv/$fileid.mfv" "$basedir/wav16/$fileid.wav"
done
