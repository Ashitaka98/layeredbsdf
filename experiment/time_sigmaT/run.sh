MTS_PROJECT_ROOT=/home/lzr/Projects/layeredbsdf

LD_LIBRARY_PATH=${MTS_PROJECT_ROOT}/dist
export LD_LIBRARY_PATH

function log() {
  echo -ne "$1" >> run.log
}

log "######################################################################\n"
log "## Experiment started at $(date)\n"
log "######################################################################\n\n"

function run_case() {
  ${MTS_PROJECT_ROOT}/dist/mitsuba dragon_GY.xml \
    -D sigmaT_0=$1 \
    -o spp64_box/dragon_GY_$1.exr
}

for sigmaT_0 in `seq 0 0.5 5`
do
  log "Timing for sigmaT_0=${sigmaT_0} ... "
  mts_output="$(run_case ${sigmaT_0})"
  log "$(echo ${mts_output} | grep -Po 'Render time: \K.*? ')\n\n"
done
