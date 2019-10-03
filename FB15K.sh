#bash run.sh train TransE FB15k 0 baseline_256 1024 256 1000 24.0 1.0 0.0001 150000 16
#
#bash run.sh train TransE FB15k 0 fake10_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10
#bash run.sh train TransE FB15k 2 fake20_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20
#bash run.sh train TransE FB15k 6 fake30_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40
#
#bash run.sh train TransE FB15k 1 LT10_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10 --method LT
#bash run.sh train TransE FB15k 2 LT20_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20 --method LT
#bash run.sh train TransE FB15k 0 LT30_256 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method LT

# bash run.sh train TransE FB15k 1 CLF30_hard 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30_256
#bash run.sh train TransE FB15k 1 CLF10_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10_256
#bash run.sh train TransE FB15k 2 CLF20_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20_256

bash run.sh train TransE FB15k 6 CLF10_hard_1 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10_256 --num 1
bash run.sh train TransE FB15k 6 CLF20_hard_1 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20_256 --num 1
bash run.sh train TransE FB15k 6 CLF30_hard_1 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30_256 --num 1

bash run.sh train TransE FB15k 6 CLF10_hard_2 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10_256 --num 2
bash run.sh train TransE FB15k 6 CLF20_hard_2 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20_256 --num 2
bash run.sh train TransE FB15k 6 CLF30_hard_2 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30_256 --num 2

bash run.sh train TransE FB15k 6 CLF10_hard_1000 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10_256 --num 1000
bash run.sh train TransE FB15k 6 CLF20_hard_1000 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20_256 --num 1000
bash run.sh train TransE FB15k 6 CLF30_hard_1000 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30_256 --num 1000


#bash run.sh train TransE FB15k 0 baseline_1 1024 1 1000 24.0 1.0 0.0001 150000 16
#bash run.sh train TransE FB15k 0 fake10_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 10
#bash run.sh train TransE FB15k 1 LT10_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 10 --method LT
#bash run.sh train TransE FB15k 1 CLF10_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10_1
#
#bash run.sh train TransE FB15k 1 fake20_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 20
#bash run.sh train TransE FB15k 1 LT20_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 20 --method LT
#bash run.sh train TransE FB15k 1 CLF20_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20_1
#
#bash run.sh train TransE FB15k 4 fake30_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 40
#bash run.sh train TransE FB15k 4 LT30_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method LT
#bash run.sh train TransE FB15k 4 CLF30_1 1024 1 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30_1
