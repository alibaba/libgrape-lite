#!/usr/bin/env bash

prefix=$1
nv=$(wc -l < "$prefix".v | xargs)
ne=$(wc -l < "$prefix".e | xargs)

echo "$nv $nv $ne" | cat - "$prefix".e > "$prefix".mtx
