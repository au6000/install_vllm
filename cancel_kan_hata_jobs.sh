#!/usr/bin/env bash
set -euo pipefail

# This format string determines the fields for awk
# 1:JOBID, 2:USER, 3:STATE, 5:REASON
SQUEUE_FMT="%.8i %.8u %.2t %.10M %.10r %.20R"

echo "================= Current squeue list ================="
squeue -o "$SQUEUE_FMT"
echo "====================================================="
echo

# WORKAROUND: Get all jobs and filter with awk, avoiding the '-u' flag.
echo "Checking for 'kan.hata' jobs in (Resources) state..."
RESOURCE_BLOCKING_JOBS=$(squeue -h -o "$SQUEUE_FMT" | awk '$2=="kan.hata" && $3=="PD" && $5=="Resources" {print $1}')

if [[ -n "$RESOURCE_BLOCKING_JOBS" ]]; then
  echo "Found jobs to cancel:"
  for jobid in $RESOURCE_BLOCKING_JOBS; do
    squeue -j "$jobid" -o "$SQUEUE_FMT" -h
    echo "Cancelling job: $jobid"
     /home/Competition2025/P09/shareP09/scripts/scancel.sh "$jobid"
  done
fi

# WORKAROUND: Find all other jobs for the user using the same awk method.
echo
echo "Checking for any remaining 'kan.hata' jobs..."
ALL_KAN_HATA_JOBS=$(squeue -h -o "$SQUEUE_FMT" | awk '$2=="kan.hata" {print $1}')

if [[ -z "$ALL_KAN_HATA_JOBS" ]]; then
  echo "No remaining 'kan.hata' jobs found."
else
  echo "Found remaining jobs to cancel:"
  for jobid in $ALL_KAN_HATA_JOBS; do
    squeue -j "$jobid" -o "$SQUEUE_FMT" -h
    echo "Cancelling job: $jobid"
     /home/Competition2025/P09/shareP09/scripts/scancel.sh "$jobid"
  done
  echo "All 'kan.hata' jobs have been processed."
fi