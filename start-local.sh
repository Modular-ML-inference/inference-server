#!/bin/bash
export FL_LOCAL_OP_DATA_FOLDER="./data"
for ((i=0; i < $1; i++));
do
  echo "Creating enabler of index $i"
  export USER_INDEX=$i
  docker-compose -p "appv$i" up --force-recreate --build -d
done
