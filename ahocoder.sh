#!/bin/sh

basedir="data/train"
nlines=${1:--0}
files=$(ls "$basedir"/wav16/*.wav | head -n $nlines);

for f in $files; do
	fileid=$(echo $f | tr '/' '\n' | tail -1 | tr '.' '\n' | head -1)
	ahocoder_64/ahocoder16_64 "$f" "$basedir/lf0/$fileid.lf0" "$basedir/mcp/$fileid.mcp" "$basedir/mfv/$fileid.mfv"
done
