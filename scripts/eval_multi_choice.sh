

cd ../
for d in piqa obqa hellaswag copa arce
do
  CUDA_VISIBLE_DEVICES=$1  python main.py \
  --dataset $d \
  --model $2 \
  --n-windows 3 \
  --n-windows 4 \
  --n-windows 6 \
  --subsample-test-set 250 \
  --n-runs 30 \
  --output-dir results_$2
done
