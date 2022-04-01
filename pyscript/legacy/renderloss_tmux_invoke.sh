#! /bin/bash
source ~/.bashrc
conda activate layeredBsdf
source /home/lzr/Projects/layeredbsdf/setpath.sh
python /home/lzr/Projects/layeredbsdf/pyscript/renderloss_data.py --begin_idx=$1 --file_num=$2