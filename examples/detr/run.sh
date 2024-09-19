#!/bin/bash
OUTDIR=${OUTDIR:-$(pwd)/outdir}
LOGFILE=${LOGFILE:-$(pwd)/console.log}
PROFILER=${1:-off}
SCRIPT=detr-ft-cppe-5.py
CMDLINE_LOG=$(pwd)/cmdline.log
OPTIONS=${OPTIONS:-}
HABANA_LOGS=${HABANA_LOGS:-$(pwd)/habana_logs}
HL_SMI_CMD="hl-smi -l 1 -f csv -Q timestamp,name,driver_version,temperature.aip,utilization.aip,memory.total,memory.free,memory.used"
HL_SMI_LOG="hlsmi.log"

mkdir -p ${HABANA_LOGS}

# Get Clean slate
cp -r ${OUTDIR} ${OUTDIR}-$(date) 2>/dev/null
rm -rf ${OUTDIR}
rm -f /root/metricslog.json.*
rm -rf $(pwd)/lightning_logs/version_*
pkill hl-smi 2>/dev/null
rm -f ${HL_SMI_LOG} 2>/dev/null

START_TIME=$(date)
${HL_SMI_CMD} 2>&1 > ${HL_SMI_LOG} &
if [[ "${PROFILER}" == "on" ]]
then   
    python ${SCRIPT} ${OPTIONS} --profile 2>/dev/null | tee ${LOGFILE}
    echo "python ${SCRIPT} ${OPTIONS} --profile 2>/dev/null" >> ${CMDLINE_LOG}
else
    python ${SCRIPT} ${OPTIONS} 2>&1 | tee ${LOGFILE}
    echo "python ${SCRIPT}  ${OPTIONS} 2>&1 | tee ${LOGFILE}" >> ${CMDLINE_LOG}
fi
pkill hl-smi

# Collect results
mkdir -p ${OUTDIR}
mv $(pwd)/lightning_logs/version_0 ${OUTDIR}/lightning_logs
mv /root/metricslog.json ${OUTDIR} 2>/dev/null
mv $(pwd)/cppe* ${OUTDIR} 2>/dev/null
mv $(pwd)/*.ckpt ${OUTDIR} 2>/dev/null
mv $(pwd)/*.jpg ${OUTDIR} 2>/dev/null
mv ${LOGFILE} ${OUTDIR}
mv ${CMDLINE_LOG} ${OUTDIR}
mkdir -p ${OUTDIR}/sources
cp *.py ${OUTDIR}/sources
cp requirements.txt ${OUTDIR}/sources
mv ${HABANA_LOGS} ${OUTDIR}
env >> ${OUTDIR}/env.log
mv ${HL_SMI_LOG} ${OUTDIR}
END_TIME=$(date)
echo START ${START_TIME} | tee ${OUTDIR}/time.log
echo STOP ${END_TIME} | tee -a ${OUTDIR}/time.log


