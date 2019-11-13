bash run.sh train TransE FB15k-237 $1 baseline 512 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE FB15k-237 $1 fake100 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100
bash run.sh train TransE FB15k-237 $1 CLF100_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100 --method clf -init ./models/TransE_FB15k-237_fake100 --num 1
bash run.sh train TransE FB15k-237 $1 CLF100_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100 --method clf -init ./models/TransE_FB15k-237_fake100 --num 1000
bash run.sh train TransE FB15k-237 $1 fake70 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70
bash run.sh train TransE FB15k-237 $1 CLF70_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70 --method clf -init ./models/TransE_FB15k-237_fake70 --num 1
bash run.sh train TransE FB15k-237 $1 CLF70_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70 --method clf -init ./models/TransE_FB15k-237_fake70 --num 1000
bash run.sh train TransE FB15k-237 $1 fake50 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 50
bash run.sh train TransE FB15k-237 $1 CLF50_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 50 --method clf -init ./models/TransE_FB15k-237_fake50 --num 1
bash run.sh train TransE FB15k-237 $1 CLF50_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 50 --method clf -init ./models/TransE_FB15k-237_fake50 --num 1000
bash run.sh train TransE FB15k-237 $1 fake30 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 30
bash run.sh train TransE FB15k-237 $1 CLF30_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 30 --method clf -init ./models/TransE_FB15k-237_fake30 --num 1
bash run.sh train TransE FB15k-237 $1 CLF30_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 30 --method clf -init ./models/TransE_FB15k-237_fake30 --num 1000

bash run.sh train TransE wn18rr $1 baseline 512 256 500 6.0 0.5 0.00005 80000 8
bash run.sh train TransE wn18rr $1 fake100 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100
bash run.sh train TransE wn18rr $1 CLF100_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100 --method clf -init ./models/TransE_wn18rr_fake100 --num 1
bash run.sh train TransE wn18rr $1 CLF100_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100 --method clf -init ./models/TransE_wn18rr_fake100 --num 1000
bash run.sh train TransE wn18rr $1 fake70 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70
bash run.sh train TransE wn18rr $1 CLF70_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake70 --num 1
bash run.sh train TransE wn18rr $1 CLF70_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake70 --num 1000
bash run.sh train TransE wn18rr $1 fake50 512 256 500 6.0 0.5 0.00005 80000 8 --fake 50
bash run.sh train TransE wn18rr $1 CLF50_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 50 --method clf -init ./models/TransE_wn18rr_fake50 --num 1
bash run.sh train TransE wn18rr $1 CLF50_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 50 --method clf -init ./models/TransE_wn18rr_fake50 --num 1000
bash run.sh train TransE wn18rr $1 fake30 512 256 500 6.0 0.5 0.00005 80000 8 --fake 30
bash run.sh train TransE wn18rr $1 CLF30_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake30 --num 1
bash run.sh train TransE wn18rr $1 CLF30_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake30 --num 1000

bash run.sh train TransE YAGO3-10 $1 baseline 512 256 500 12.0 0.5 0.001 100000 8
bash run.sh train TransE YAGO3-10 $1 fake100 512 256 500 12.0 0.5 0.001 100000 8 --fake 100
bash run.sh train TransE YAGO3-10 $1 CLF100_soft 512 256 500 12.0 0.5 0.001 100000 8 --fake 100 --method clf -init ./models/TransE_YAGO3-10_fake100 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF100_hard 512 256 500 12.0 0.5 0.001 100000 8 --fake 100 --method clf -init ./models/TransE_YAGO3-10_fake100 --num 1000
bash run.sh train TransE YAGO3-10 $1 fake70 512 256 500 12.0 0.5 0.001 100000 8 --fake 70
bash run.sh train TransE YAGO3-10 $1 CLF70_soft 512 256 500 12.0 0.5 0.001 100000 8 --fake 70 --method clf -init ./models/TransE_YAGO3-10_fake70 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF70_hard 512 256 500 12.0 0.5 0.001 100000 8 --fake 70 --method clf -init ./models/TransE_YAGO3-10_fake70 --num 1000
bash run.sh train TransE YAGO3-10 $1 fake50 512 256 500 12.0 0.5 0.001 100000 8 --fake 50
bash run.sh train TransE YAGO3-10 $1 CLF50_soft 512 256 500 12.0 0.5 0.001 100000 8 --fake 50 --method clf -init ./models/TransE_YAGO3-10_fake50 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF50_hard 512 256 500 12.0 0.5 0.001 100000 8 --fake 50 --method clf -init ./models/TransE_YAGO3-10_fake50 --num 1000
bash run.sh train TransE YAGO3-10 $1 fake30 512 256 500 12.0 0.5 0.001 100000 8 --fake 30
bash run.sh train TransE YAGO3-10 $1 CLF30_soft 512 256 500 12.0 0.5 0.001 100000 8 --fake 30 --method clf -init ./models/TransE_YAGO3-10_fake30 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF30_hard 512 256 500 12.0 0.5 0.001 100000 8 --fake 30 --method clf -init ./models/TransE_YAGO3-10_fake30 --num 1000