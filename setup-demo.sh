#!/bin/bash
for ((i=0; i < $1; i++));
do
  echo "Insert correct demo data to enabler $i"
  docker cp "./demo-data/train_x_$i.npy" "appv$i-local_operations-1:data/x_train.npy"
  docker cp "./demo-data/train_y_$i.npy" "appv$i-local_operations-1:data/y_train.npy"
done
