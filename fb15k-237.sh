#bash run.sh train TransE FB15k-237 2 baseline 1024 256 1000 9.0 1.0 0.00005 100000 16
#
#bash run.sh train TransE FB15k-237 2 fake10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10
#bash run.sh train TransE FB15k-237 2 fake20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20
#bash run.sh train TransE FB15k-237 2 fake40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40
#
#
#bash run.sh train TransE FB15k-237 2 LT10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method CLT
#bash run.sh train TransE FB15k-237 2 LT20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method CLT
#bash run.sh train TransE FB15k-237 2 LT40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method CLT

bash run.sh train TransE FB15k-237 5 CLF10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method clf -init ./models/TransE_FB15k-237_fake10
bash run.sh train TransE FB15k-237 5 CLF20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method clf -init ./models/TransE_FB15k-237_fake20
bash run.sh train TransE FB15k-237 5 CLF40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40
