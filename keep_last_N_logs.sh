#!/bin/bash

echo logdir=$1, N=$2
logdir=$1
n_keep=$2

# remove these temp files
rm -r $logdir/mean_regressors

model_logs=$(/bin/ls -t $logdir/*/version*/config.yaml)
#echo model_logs: $model_logs
#return 0 || exit 0
#ls -ltr $model_logs
n_total=$(echo $model_logs | wc -w)
n_drop=$(( n_total-n_keep ))

echo counted $n_total existing models!
echo dropping $n_drop old models

model_logs=$(echo $model_logs | tr ' ' '\n' | tail -n $n_drop)
echo model_logs:
ls -ltr $model_logs

# remove all old model log directories
model_logs=$(dirname $model_logs)
rm -r $model_logs
