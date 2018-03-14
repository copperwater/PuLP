#!/bin/bash

if [ -z $1 ]; then
  echo Usage: $0 adjacency-file
  exit 1
fi

if [ ! -e $1 ]; then
  echo Adjacency file $1 does not exist.
  exit 1
fi

file=$1
header=$(head -n 1 $file)
read nverts nedges mode nweights <<< $header
words=$(wc -w < $file)

echo File has $words space-delimited words. 4 should be from header.
totalweights=$(( nweights * nverts ))
echo File should have $totalweights weights, in groups of $nweights at the start of each line.
echo Header says file has $nedges edges, *2 = $(( nedges * 2 )) edges.
actualedges=$(( words - (totalweights + 4) ))
echo File actually contains $actualedges edges.
if [[ $actualedges != $(( $nedges * 2 )) ]]; then
  echo Error: number of edges does not match header!
else
  echo Number of edges matches and the file appears to be valid.
fi
