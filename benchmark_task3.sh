#!/bin/bash
EXEC=./radiator_gpu_task3
N=15360 M=15360 P=1000

declare -a Bx=(16 32 64 8)
declare -a By=(16 8 4 32)

echo "bx,by,Propagate_ms,Average_ms,Total_ms" > bench.csv

for bx in "${Bx[@]}"; do
  for by in "${By[@]}"; do
    if (( N % bx != 0 || M % by != 0 )); then
      continue
    fi

    out=$($EXEC -c -t -n $N -m $M -p $P -bx $bx -by $by | \
      awk '
        /GPU propagate/ {prop=$3}
        /GPU average/   {avg=$3}
        END {
          total = prop + avg
          print prop "," avg "," total
        }
      ')
    
    echo "$bx,$by,$out" >> bench.csv
  done
done

echo "Done. Results in bench.csv"
