#!/bin/bash

logdir=$1
n_keep=$2

unalias rm 2> /dev/null
echo logdir=$logdir, N=$n_keep, pwd=$(pwd)

# remove these temp files
rm -r $logdir/UQ_regressors 2> /dev/null
rm -r $logdir/mean_regressors 2> /dev/null
rm -r $logdir/InteractiveJob 2> /dev/null

model_logs=$(/bin/ls -t $logdir/*/version*/config.yaml)
n_total=$(echo $model_logs | wc -w)
n_drop=$(( n_total-n_keep ))

echo counted $n_total existing models!
echo dropping $n_drop old models

if ((n_total<=n_keep)); then
    echo nothing to do... exiting
    return 0 || exit 0
fi

model_logs=$(echo $model_logs | tr ' ' '\n' | tail -n $n_drop)
echo model_logs:
ls -ltr $model_logs

# remove all old model log directories
model_logs=$(dirname $model_logs)
rm -r $model_logs

# also remove all newly empty experiment folders
for fn in $logdir/* ; do
    [[ -d $fn && -z $(/bin/ls -A $fn) ]] && rm -r $fn
done
