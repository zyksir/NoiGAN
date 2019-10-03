#bash run.sh train TransE wn18 2 baseline_256 512 1024 500 12.0 0.5 0.0001 80000 8
#
#bash run.sh train TransE wn18 0 fake10_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10
#bash run.sh train TransE wn18 1 fake20_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20
#bash run.sh train TransE wn18 4 fake30_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40
#
#bash run.sh train TransE wn18 3 LT10_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method LT
#bash run.sh train TransE wn18 1 LT20_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method LT
#bash run.sh train TransE wn18 4 LT30_256 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method LT

#bash run.sh train TransE wn18 0 CLF10_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10_256
#bash run.sh train TransE wn18 0 CLF20_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20_256
#bash run.sh train TransE wn18 0 CLF30_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake30_256

bash run.sh train TransE wn18 0 CLF10_hard_1 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10_256 --num 1
bash run.sh train TransE wn18 0 CLF20_hard_1 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20_256 --num 1
bash run.sh train TransE wn18 0 CLF30_hard_1 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake30_256 --num 1

bash run.sh train TransE wn18 0 CLF10_hard_2 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10_256 --num 2
bash run.sh train TransE wn18 0 CLF20_hard_2 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20_256 --num 2
bash run.sh train TransE wn18 0 CLF30_hard_2 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake30_256 --num 2

bash run.sh train TransE wn18 0 CLF10_hard_1000 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10_256 --num 1000
bash run.sh train TransE wn18 0 CLF20_hard_1000 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20_256 --num 1000
bash run.sh train TransE wn18 0 CLF30_hard_1000 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake30_256 --num 1000

#bash run.sh train TransE wn18 4 baseline_1 512 1 500 12.0 0.5 0.0001 80000 8
#bash run.sh train TransE wn18 4 fake10_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 10
#bash run.sh train TransE wn18 5 LT10_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 10 --method LT
#bash run.sh train TransE wn18 3 CLF10_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10_1
#
#bash run.sh train TransE wn18 2 fake20_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 20
#bash run.sh train TransE wn18 2 LT20_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 20 --method LT
#bash run.sh train TransE wn18 2 CLF20_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20_1
#
#bash run.sh train TransE wn18 4 fake30_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 40
#bash run.sh train TransE wn18 5 LT30_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 40 --method LT
#bash run.sh train TransE wn18 7 CLF30_1 512 1 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake30_1


