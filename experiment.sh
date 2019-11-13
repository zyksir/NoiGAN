bash run.sh train DistMult FB15k-237 $1 baseline 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult FB15k-237 $1 fake100 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 100
bash run.sh train DistMult FB15k-237 $1 CLF100_soft 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 100 --method clf -init ./models/DistMult_FB15k-237_fake100  --num 1
bash run.sh train DistMult FB15k-237 $1 CLF100_hard 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 100 --method clf -init ./models/DistMult_FB15k-237_fake100  --num 1000
bash run.sh train DistMult FB15k-237 $1 fakePath100 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake Path100
bash run.sh train DistMult FB15k-237 $1 CLFPath100_soft 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake Path100 --method clf -init ./models/DistMult_FB15k-237_fakePath100  --num 1
bash run.sh train DistMult FB15k-237 $1 CLFPath100_hard 1024 256 500 200.0 1.0 0.001 100000 16 -r 0.00001 --fake Path100 --method clf -init ./models/DistMult_FB15k-237_fakePath100  --num 1000

bash run.sh train DistMult wn18rr $1 baseline 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005
bash run.sh train DistMult wn18rr $1 fake100 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 100
bash run.sh train DistMult wn18rr $1 CLF100_soft 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 100 --method clf -init ./models/DistMult_wn18rr_fake100  --num 1
bash run.sh train DistMult wn18rr $1 CLF100_hard 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 100 --method clf -init ./models/DistMult_wn18rr_fake100  --num 1000
bash run.sh train DistMult wn18rr $1 fakePath100 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake Path100
bash run.sh train DistMult wn18rr $1 CLFPath100_soft 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake Path100 --method clf -init ./models/DistMult_wn18rr_fakePath100  --num 1
bash run.sh train DistMult wn18rr $1 CLFPath100_hard 512 1024 500 200.0 1.0 0.002 80000 8 -r 0.000005 --fake Path100 --method clf -init ./models/DistMult_wn18rr_fakePath100  --num 1000


bash run.sh train RotatE FB15k-237 $1 baseline 1024 256 250 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE FB15k-237 $1 fake100 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 100
bash run.sh train RotatE FB15k-237 $1 CLF100_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 100 --method clf -init ./models/RotatE_FB15k-237_fake100  --num 1
bash run.sh train RotatE FB15k-237 $1 CLF100_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 100 --method clf -init ./models/RotatE_FB15k-237_fake100  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath100 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100
bash run.sh train RotatE FB15k-237 $1 CLFPath100_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100 --method clf -init ./models/RotatE_FB15k-237_fakePath100  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath100_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100 --method clf -init ./models/RotatE_FB15k-237_fakePath100  --num 1000

bash run.sh train RotatE FB15k-237 $1 fake70 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 70
bash run.sh train RotatE FB15k-237 $1 CLF70_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 70 --method clf -init ./models/RotatE_FB15k-237_fake70  --num 1
bash run.sh train RotatE FB15k-237 $1 CLF70_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 70 --method clf -init ./models/RotatE_FB15k-237_fake70  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath70 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70
bash run.sh train RotatE FB15k-237 $1 CLFPath70_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70 --method clf -init ./models/RotatE_FB15k-237_fakePath70  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath70_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70 --method clf -init ./models/RotatE_FB15k-237_fakePath70  --num 1000

bash run.sh train RotatE FB15k-237 $1 fake50 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 50
bash run.sh train RotatE FB15k-237 $1 CLF50_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 50 --method clf -init ./models/RotatE_FB15k-237_fake50  --num 1
bash run.sh train RotatE FB15k-237 $1 CLF50_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 50 --method clf -init ./models/RotatE_FB15k-237_fake50  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath50 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path50
bash run.sh train RotatE FB15k-237 $1 CLFPath50_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path50 --method clf -init ./models/RotatE_FB15k-237_fakePath50  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath50_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path50 --method clf -init ./models/RotatE_FB15k-237_fakePath50  --num 1000

bash run.sh train RotatE FB15k-237 $1 fake30 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 30
bash run.sh train RotatE FB15k-237 $1 CLF30_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 30 --method clf -init ./models/RotatE_FB15k-237_fake30  --num 1
bash run.sh train RotatE FB15k-237 $1 CLF30_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 30 --method clf -init ./models/RotatE_FB15k-237_fake30  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath30 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path30
bash run.sh train RotatE FB15k-237 $1 CLFPath30_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path30 --method clf -init ./models/RotatE_FB15k-237_fakePath30  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath30_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path30 --method clf -init ./models/RotatE_FB15k-237_fakePath30  --num 1000


bash run.sh train RotatE wn18rr $1 baseline 512 1024 250 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE wn18rr $1 fake100 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100
bash run.sh train RotatE wn18rr $1 CLF100_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100 --method clf -init ./models/RotatE_wn18rr_fake100  --num 1
bash run.sh train RotatE wn18rr $1 CLF100_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100 --method clf -init ./models/RotatE_wn18rr_fake100  --num 1000
bash run.sh train RotatE wn18rr $1 fakePath100 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path100
bash run.sh train RotatE wn18rr $1 CLFPath100_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path100 --method clf -init ./models/RotatE_wn18rr_fakePath100  --num 1
bash run.sh train RotatE wn18rr $1 CLFPath100_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path100 --method clf -init ./models/RotatE_wn18rr_fakePath100  --num 1000

bash run.sh train RotatE wn18rr $1 fake70 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70
bash run.sh train RotatE wn18rr $1 CLF70_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70 --method clf -init ./models/RotatE_wn18rr_fake70  --num 1
bash run.sh train RotatE wn18rr $1 CLF70_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70 --method clf -init ./models/RotatE_wn18rr_fake70  --num 1000
bash run.sh train RotatE wn18rr $1 fakePath70 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path70
bash run.sh train RotatE wn18rr $1 CLFPath70_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path70 --method clf -init ./models/RotatE_wn18rr_fakePath70  --num 1
bash run.sh train RotatE wn18rr $1 CLFPath70_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path70 --method clf -init ./models/RotatE_wn18rr_fakePath70  --num 1000

bash run.sh train RotatE wn18rr $1 fake30 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 30
bash run.sh train RotatE wn18rr $1 CLF30_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 30 --method clf -init ./models/RotatE_wn18rr_fake30  --num 1
bash run.sh train RotatE wn18rr $1 CLF30_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 30 --method clf -init ./models/RotatE_wn18rr_fake30  --num 1000
bash run.sh train RotatE wn18rr $1 fakePath30 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path30
bash run.sh train RotatE wn18rr $1 CLFPath30_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path30 --method clf -init ./models/RotatE_wn18rr_fakePath30  --num 1
bash run.sh train RotatE wn18rr $1 CLFPath30_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path30 --method clf -init ./models/RotatE_wn18rr_fakePath30  --num 1000

bash run.sh train RotatE wn18rr $1 fake50 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 50
bash run.sh train RotatE wn18rr $1 CLF50_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 50 --method clf -init ./models/RotatE_wn18rr_fake50  --num 1
bash run.sh train RotatE wn18rr $1 CLF50_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 50 --method clf -init ./models/RotatE_wn18rr_fake50  --num 1000
bash run.sh train RotatE wn18rr $1 fakePath50 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path50
bash run.sh train RotatE wn18rr $1 CLFPath50_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path50 --method clf -init ./models/RotatE_wn18rr_fakePath50  --num 1
bash run.sh train RotatE wn18rr $1 CLFPath50_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake Path50 --method clf -init ./models/RotatE_wn18rr_fakePath50  --num 1000


bash run.sh train RotatE YAGO3-10 $1 baseline 1024 400 50 24.0 1.0 0.0002 100000 4 -de
bash run.sh train RotatE YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 100
bash run.sh train RotatE YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 100 --method clf -init ./models/RotatE_YAGO3-10_fake100  --num 1
bash run.sh train RotatE YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 100 --method clf -init ./models/RotatE_YAGO3-10_fake100  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fakePath100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path100
bash run.sh train RotatE YAGO3-10 $1 fakePath100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path100 --method clf -init ./models/RotatE_YAGO3-10_fakePath100  --num 1
bash run.sh train RotatE YAGO3-10 $1 fakePath100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path100 --method clf -init ./models/RotatE_YAGO3-10_fakePath100  --num 1000

bash run.sh train RotatE YAGO3-10 $1 fake70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 70
bash run.sh train RotatE YAGO3-10 $1 fake70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 70 --method clf -init ./models/RotatE_YAGO3-10_fake70  --num 1
bash run.sh train RotatE YAGO3-10 $1 fake70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 70 --method clf -init ./models/RotatE_YAGO3-10_fake70  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fakePath70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path70
bash run.sh train RotatE YAGO3-10 $1 fakePath70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path70 --method clf -init ./models/RotatE_YAGO3-10_fakePath70  --num 1
bash run.sh train RotatE YAGO3-10 $1 fakePath70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path70 --method clf -init ./models/RotatE_YAGO3-10_fakePath70  --num 1000

bash run.sh train RotatE YAGO3-10 $1 fake50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 50
bash run.sh train RotatE YAGO3-10 $1 fake50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 50 --method clf -init ./models/RotatE_YAGO3-10_fake50  --num 1
bash run.sh train RotatE YAGO3-10 $1 fake50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 50 --method clf -init ./models/RotatE_YAGO3-10_fake50  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fakePath50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path50
bash run.sh train RotatE YAGO3-10 $1 fakePath50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path50 --method clf -init ./models/RotatE_YAGO3-10_fakePath50  --num 1
bash run.sh train RotatE YAGO3-10 $1 fakePath50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path50 --method clf -init ./models/RotatE_YAGO3-10_fakePath50  --num 1000

bash run.sh train RotatE YAGO3-10 $1 fake30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 30
bash run.sh train RotatE YAGO3-10 $1 fake30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 30 --method clf -init ./models/RotatE_YAGO3-10_fake30  --num 1
bash run.sh train RotatE YAGO3-10 $1 fake30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 30 --method clf -init ./models/RotatE_YAGO3-10_fake30  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fakePath30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path30
bash run.sh train RotatE YAGO3-10 $1 fakePath30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path30 --method clf -init ./models/RotatE_YAGO3-10_fakePath30  --num 1
bash run.sh train RotatE YAGO3-10 $1 fakePath30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake Path30 --method clf -init ./models/RotatE_YAGO3-10_fakePath30  --num 1000


#bash run.sh train TransE wn18rr $1 baseline 512 1024 500 6.0 0.5 0.00005 80000 8
#bash run.sh train TransE wn18rr $1 fake10 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10
#bash run.sh train TransE wn18rr $1 LT10 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10 --method LT
#bash run.sh train TransE wn18rr $1 CLF10_soft 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10 --method clf -init ./models/TransE_wn18rr_fake10  --num 1
#bash run.sh train TransE wn18rr $1 CLF10_hard 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10 --method clf -init ./models/TransE_wn18rr_fake10  --num 1000
#
#bash run.sh train TransE wn18rr $1 fake20 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20
#bash run.sh train TransE wn18rr $1 LT20 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20 --method LT
#bash run.sh train TransE wn18rr $1 CLF20_soft 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20 --method clf -init ./models/TransE_wn18rr_fake20  --num 1
#bash run.sh train TransE wn18rr $1 CLF20_hard 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20 --method clf -init ./models/TransE_wn18rr_fake20  --num 1000
#
#bash run.sh train TransE wn18rr $1 fake40 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40
#bash run.sh train TransE wn18rr $1 LT40 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40 --method LT
#bash run.sh train TransE wn18rr $1 CLF40_soft 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40 --method clf -init ./models/TransE_wn18rr_fake40  --num 1
#bash run.sh train TransE wn18rr $1 CLF40_hard 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40 --method clf -init ./models/TransE_wn18rr_fake40  --num 1000
#


bash run.sh train TransE FB15k $1 baseline 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k $1 fake10 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10
bash run.sh train TransE FB15k $1 fake20 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20
bash run.sh train TransE FB15k $1 fake30 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40
bash run.sh train TransE FB15k $1 LT10 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 10 --method LT
bash run.sh train TransE FB15k $1 LT20 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 20 --method LT
bash run.sh train TransE FB15k $1 LT30 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method LT
bash run.sh train TransE FB15k $1 CLF10_soft 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10  --num 1
bash run.sh train TransE FB15k $1 CLF20_soft 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20  --num 1
bash run.sh train TransE FB15k $1 CLF40_soft 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30  --num 1
bash run.sh train TransE FB15k $1 CLF10_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 10 --method clf -init ./models/TransE_FB15k_fake10  --num 1000
bash run.sh train TransE FB15k $1 CLF20_hard 1024 256 1000 24.0 1.0 0.00001 150000 16 --fake 20 --method clf -init ./models/TransE_FB15k_fake20  --num 1000
bash run.sh train TransE FB15k $1 CLF40_hard 1024 256 1000 24.0 1.0 0.0001 150000 16 --fake 40 --method clf -init ./models/TransE_FB15k_fake30  --num 1000
#
#
#bash run.sh train TransE FB15k-237 $1 baseline 1024 256 1000 9.0 1.0 0.00005 100000 16
#bash run.sh train TransE FB15k-237 $1 fake10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10
#bash run.sh train TransE FB15k-237 $1 fake20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20
#bash run.sh train TransE FB15k-237 $1 fake40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40
#bash run.sh train TransE FB15k-237 $1 LT10 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method CLT
#bash run.sh train TransE FB15k-237 $1 LT20 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method CLT
#bash run.sh train TransE FB15k-237 $1 LT40 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method CLT
#bash run.sh train TransE FB15k-237 $1 CLF10_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method clf -init ./models/TransE_FB15k-237_fake10  --num 1
#bash run.sh train TransE FB15k-237 $1 CLF20_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method clf -init ./models/TransE_FB15k-237_fake20  --num 1
#bash run.sh train TransE FB15k-237 $1 CLF40_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40  --num 1
#bash run.sh train TransE FB15k-237 $1 CLF10_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 10 --method clf -init ./models/TransE_FB15k-237_fake10  --num 1000
#bash run.sh train TransE FB15k-237 $1 CLF20_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 20 --method clf -init ./models/TransE_FB15k-237_fake20  --num 1000
#bash run.sh train TransE FB15k-237 $1 CLF40_soft 1024 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40  --num 1000
#
#
#bash run.sh train TransE wn18 $1 baseline56 512 1024 500 12.0 0.5 0.0001 80000 8
#bash run.sh train TransE wn18 $1 fake10 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10
#bash run.sh train TransE wn18 $1 fake20 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20
#bash run.sh train TransE wn18 $1 fake40 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40
#bash run.sh train TransE wn18 $1 LT10 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method LT
#bash run.sh train TransE wn18 $1 LT20 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method LT
#bash run.sh train TransE wn18 $1 LT30 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method LT
#bash run.sh train TransE wn18 $1 CLF10_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10 --num 1
#bash run.sh train TransE wn18 $1 CLF20_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20 --num 1
#bash run.sh train TransE wn18 $1 CLF40_soft 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake40 --num 1
#bash run.sh train TransE wn18 $1 CLF10_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 10 --method clf -init ./models/TransE_wn18_fake10 --num 1000
#bash run.sh train TransE wn18 $1 CLF20_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 20 --method clf -init ./models/TransE_wn18_fake20 --num 1000
#bash run.sh train TransE wn18 $1 CLF40_hard 512 1024 500 12.0 0.5 0.0001 80000 8 --fake 40 --method clf -init ./models/TransE_wn18_fake40 --num 1000
#
#
#bash run.sh train TransE YAGO3-10 $1 baseline2 512 1024 250 12.0 0.5 0.001 100000 8
#bash run.sh train TransE YAGO3-10 $1 fake10 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10
#bash run.sh train TransE YAGO3-10 $1 fake20 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20
#bash run.sh train TransE YAGO3-10 $1 fake40 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40
#bash run.sh train TransE YAGO3-10 $1 LT10 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method LT
#bash run.sh train TransE YAGO3-10 $1 LT20 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method LT
#bash run.sh train TransE YAGO3-10 $1 LT40 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method LT
#bash run.sh train TransE YAGO3-10 $1 CLF10_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method clf -init ./models/TransE_YAGO3-10_fake10 --num 1
#bash run.sh train TransE YAGO3-10 $1 CLF20_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method clf -init ./models/TransE_YAGO3-10_fake20 --num 1
#bash run.sh train TransE YAGO3-10 $1 CLF40_soft 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1
#bash run.sh train TransE YAGO3-10 $1 CLF10_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 10 --method clf -init ./models/TransE_YAGO3-10_fake10 --num 1000
#bash run.sh train TransE YAGO3-10 $1 CLF20_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 20 --method clf -init ./models/TransE_YAGO3-10_fake20 --num 1000
#bash run.sh train TransE YAGO3-10 $1 CLF40_hard 512 1024 250 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1000


#bash run.sh train DistMult wn18 5 baseline 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001
#bash run.sh train ComplEx wn18 5 baseline 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
#bash run.sh train DistMult wn18 5 fake10 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001 --fake 10
#bash run.sh train ComplEx wn18 5 fake10 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 --fake 10
bash run.sh train ComplEx wn18 4 fake10_half 512 1024 500 200.0 1.0 0.001 20000 8 -de -dr -r 0.00001 --fake 10
bash run.sh train ComplEx FB15k 4 fake10_half 1024 256 500 500.0 1.0 0.001 30000 16 -de -dr -r 0.000002 --fake 10

# bash run.sh train DistMult FB15k-237 7 fake10 1024 256 1000 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 10
bash run.sh train TransE FB15k-237 6 KBGAN_fake10 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_fake10 --gen_init ./models/DistMult_FB15k-237_fake10 --fake 10
bash run.sh train TransE FB15k-237 6 KBGAN 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_baseline --gen_init ./models/DistMult_FB15k-237_baseline --fake 10

