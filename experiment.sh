bash run.sh train TransE FB15k 0 baseline 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k 0 fake10 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10
bash run.sh train TransE FB15k 0 fake20 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20
bash run.sh train TransE FB15k 0 fake30 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40
bash run.sh train TransE FB15k 0 LT10 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10 --method LT
bash run.sh train TransE FB15k 0 LT20 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20 --method LT
bash run.sh train TransE FB15k 0 LT30 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method LT
bash run.sh train TransE FB15k 0 CLF10_soft 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10  --num 1
bash run.sh train TransE FB15k 0 CLF20_soft 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20  --num 1
bash run.sh train TransE FB15k 0 CLF40_soft 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30  --num 1
bash run.sh train TransE FB15k 0 CLF10_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10  --num 1000
bash run.sh train TransE FB15k 0 CLF20_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20  --num 1000
bash run.sh train TransE FB15k 0 CLF40_hard 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30  --num 1000


bash run.sh train TransE FB15k-237 0 baseline 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE FB15k-237 0 fake10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10
bash run.sh train TransE FB15k-237 0 fake20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20
bash run.sh train TransE FB15k-237 0 fake40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40
bash run.sh train TransE FB15k-237 0 LT10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method CLT
bash run.sh train TransE FB15k-237 0 LT20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method CLT
bash run.sh train TransE FB15k-237 0 LT40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method CLT
bash run.sh train TransE FB15k-237 0 CLF10_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method clf -init ./models/TransE_FB15k-237_fake10  --num 1
bash run.sh train TransE FB15k-237 0 CLF20_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method clf -init ./models/TransE_FB15k-237_fake20  --num 1
bash run.sh train TransE FB15k-237 0 CLF40_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40  --num 1
bash run.sh train TransE FB15k-237 0 CLF10_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method clf -init ./models/TransE_FB15k-237_fake10  --num 1000
bash run.sh train TransE FB15k-237 0 CLF20_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method clf -init ./models/TransE_FB15k-237_fake20  --num 1000
bash run.sh train TransE FB15k-237 0 CLF40_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40  --num 1000


bash run.sh train TransE wn18 0 baseline56 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train TransE wn18 0 fake10 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10
bash run.sh train TransE wn18 0 fake20 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20
bash run.sh train TransE wn18 0 fake40 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40
bash run.sh train TransE wn18 0 LT10 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method LT
bash run.sh train TransE wn18 0 LT20 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method LT
bash run.sh train TransE wn18 0 LT30 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method LT
bash run.sh train TransE wn18 0 CLF10_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10 --num 1
bash run.sh train TransE wn18 0 CLF20_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20 --num 1
bash run.sh train TransE wn18 0 CLF40_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake40 --num 1
bash run.sh train TransE wn18 0 CLF10_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10 --num 1000
bash run.sh train TransE wn18 0 CLF20_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20 --num 1000
bash run.sh train TransE wn18 0 CLF40_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake40 --num 1000


bash run.sh train TransE YAGO3-10 0 baseline2 512 1024 250 12.0 0.5 0.001 100000 8
bash run.sh train TransE YAGO3-10 0 fake10 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10
bash run.sh train TransE YAGO3-10 0 fake20 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20
bash run.sh train TransE YAGO3-10 0 fake40 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40
bash run.sh train TransE YAGO3-10 0 LT10 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method LT
bash run.sh train TransE YAGO3-10 0 LT20 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method LT
bash run.sh train TransE YAGO3-10 0 LT40 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method LT
bash run.sh train TransE YAGO3-10 0 CLF10_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method clf -init ./models/TransE_YAGO3-10_fake10 --num 1
bash run.sh train TransE YAGO3-10 0 CLF20_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method clf -init ./models/TransE_YAGO3-10_fake20 --num 1
bash run.sh train TransE YAGO3-10 0 CLF40_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1
bash run.sh train TransE YAGO3-10 0 CLF10_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method clf -init ./models/TransE_YAGO3-10_fake10 --num 1000
bash run.sh train TransE YAGO3-10 0 CLF20_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method clf -init ./models/TransE_YAGO3-10_fake20 --num 1000
bash run.sh train TransE YAGO3-10 0 CLF40_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1000

bash run.sh train DistMult FB15k-237 5 baseline 1024 256 1000 200.0 1.0 0.001 100000 16 -r 0.00001