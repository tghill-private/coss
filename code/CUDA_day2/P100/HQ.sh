#!/bin/bash


if test $# -eq 0
 then
 echo "Syntax: HQ.sh  N_threads"
 exit
 fi

for ((i=0; i<$1; i++))
 do
 echo $i
 ./primes_HQ &>out &
 done

wait
