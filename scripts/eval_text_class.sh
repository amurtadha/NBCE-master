

cd ../

for d in   sst2 sst5 cr cb subj trec rte dbpedia clinic150 banking77 nlu nluscenario trecfine
do
 CUDA_VISIBLE_DEVICES=$1  python main.py \
  --dataset $d \
  --model $2 \
  --n-windows 3 \
  --n-windows 6 \
  --n-windows 9 \
  --subsample-test-set 250 \
  --n-runs 30 \
  --output-dir results_$2
done
