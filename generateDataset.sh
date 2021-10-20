#! /bin/bash
session_prefix=$1
core_num=$2
i=1
while (($i<=$core_num))
do
    tmux new-session -d -s $session_prefix$i "bash --init-file /home/lzr/Projects/layeredbsdf/pyscript/tmux_invoke.sh"
    let i++
done