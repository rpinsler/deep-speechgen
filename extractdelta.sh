#!/bin/sh

basedir="data/train"
windir="HTS/data/win"
nlines=${1:--0}
files=$(ls "$basedir"/lf0/*.lf0 | head -n $nlines);

for f in $files; do
	fileid=$(echo $f | tr '/' '\n' | tail -1 | tr '.' '\n' | head -1)
	HTS/data/scripts/window.pl 1 "$basedir/lf0/$fileid.lf0" "$windir/lf0.win1" "$windir/lf0.win2" "$windir/lf0.win3" > "$basedir/lf0/$fileid.lf0d"
	HTS/data/scripts/window.pl 1 "$basedir/mfv/$fileid.mfv" "$windir/lf0.win1" "$windir/lf0.win2" "$windir/lf0.win3" > "$basedir/mfv/$fileid.mfvd"
	HTS/data/scripts/window.pl 40 "$basedir/mcp/$fileid.mcp" "$windir/mgc.win1" "$windir/mgc.win2" "$windir/mgc.win3" > "$basedir/mcp/$fileid.mcpd"
done
