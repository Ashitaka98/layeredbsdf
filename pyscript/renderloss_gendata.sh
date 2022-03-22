#! /bin/bash
session_prefix=$1
core_num=$2
total_files_num=$3
i=1
let file_num=$total_files_num/$core_num+1
begin_idx=0
while (($i<=$core_num))
do
    echo ???
    tmux new-session -d -s $session_prefix$i "bash --init-file <(echo source ~/.bashrc \&\& \
conda activate layeredBsdf \&\& \
source /home/lzr/Projects/layeredbsdf/setpath.sh \&\& \
python /home/lzr/Projects/layeredbsdf/pyscript/renderloss_data.py --begin_idx=$begin_idx --file_num=$file_num)"
    let i++
    let begin_idx=$begin_idx+$file_num
done