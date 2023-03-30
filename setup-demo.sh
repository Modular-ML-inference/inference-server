#!/bin/bash
for ((i=0; i < $1; i++));
do
  echo "Insert correct demo data to enabler $i"
  docker cp "./demo-data/demo-data/train_x_$i.npy" "appv$i""_local_operations_1:data/x_train.npy"
  docker cp "./demo-data/demo-data/train_y_$i.npy" "appv$i""_local_operations_1:data/y_train.npy"
  docker cp "./demo-data/demo-data/test_x_$i.npy" "appv$i""_local_operations_1:data/x_test.npy"
  docker cp "./demo-data/demo-data/test_y_$i.npy" "appv$i""_local_operations_1:data/y_test.npy"
done
