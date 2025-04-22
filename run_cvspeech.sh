
LOGFILE="cvspeech_pipeline.log"

# Group all steps so the entire block is backgrounded and logged
echo "Starting cvspeech pipeline at $(date)" >> "$LOGFILE"
{
    python dl_and_preprop_dataset.py --dataset=cvspeech
    python train-text2mel.py        --dataset=cvspeech
    python train-ssrn.py            --dataset=cvspeech
    python synthesize.py            --dataset=cvspeech
    echo "Finished cvspeech pipeline at $(date)"
} >> "$LOGFILE" 2>&1 &

# PID of the background job
echo "Pipeline PID: $!" >> "$LOGFILE"
echo "cvspeech pipeline started in background (see $LOGFILE for logs)"
