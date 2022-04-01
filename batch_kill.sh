#! /bin/bash
session_prefix=$1
core_num=$2
i=1
while (($i<=$core_num))
do
    tmux kill-session -t $session_prefix$i
    let i++
done