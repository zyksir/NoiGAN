bash run.sh train TransE FB15k-237 $1 baseline 512 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE FB15k-237 $1 baseline_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --method clf -init ./models/TransE_FB15k-237_baseline --num 1
bash run.sh train TransE FB15k-237 $1 baseline_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --method clf -init ./models/TransE_FB15k-237_baseline --num 1000
bash run.sh train TransE FB15k-237 $1 fake100 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100
bash run.sh train TransE FB15k-237 $1 CLF100_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100 --method clf -init ./models/TransE_FB15k-237_fake100 --num 1
bash run.sh train TransE FB15k-237 $1 CLF100_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 100 --method clf -init ./models/TransE_FB15k-237_fake100 --num 1000
bash run.sh train TransE FB15k-237 $1 fake70 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70
bash run.sh train TransE FB15k-237 $1 CLF70_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70 --method clf -init ./models/TransE_FB15k-237_fake70 --num 1
bash run.sh train TransE FB15k-237 $1 CLF70_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 70 --method clf -init ./models/TransE_FB15k-237_fake70 --num 1000
bash run.sh train TransE FB15k-237 $1 fake40 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 40
bash run.sh train TransE FB15k-237 $1 CLF40_soft 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40 --num 1
bash run.sh train TransE FB15k-237 $1 CLF40_hard 512 256 1000 9.0 1.0 0.00005 100000 16 --fake 40 --method clf -init ./models/TransE_FB15k-237_fake40 --num 1000

bash run.sh train TransE wn18rr $1 baseline 512 256 500 6.0 0.5 0.00005 80000 8
bash run.sh train TransE wn18rr $1 baseline_soft 512 256 500 6.0 0.5 0.00005 80000 8 --method clf -init ./models/TransE_wn18rr_baseline --num 1
bash run.sh train TransE wn18rr $1 baseline_hard 512 256 500 6.0 0.5 0.00005 80000 8 --method clf -init ./models/TransE_wn18rr_baseline --num 1000
bash run.sh train TransE wn18rr $1 fake100 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100
bash run.sh train TransE wn18rr $1 CLF100_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100 --method clf -init ./models/TransE_wn18rr_fake100 --num 1
bash run.sh train TransE wn18rr $1 CLF100_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100 --method clf -init ./models/TransE_wn18rr_fake100 --num 1000
bash run.sh train TransE wn18rr $1 fake70 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70
bash run.sh train TransE wn18rr $1 CLF70_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake70 --num 1
bash run.sh train TransE wn18rr $1 CLF70_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method clf -init ./models/TransE_wn18rr_fake70 --num 1000
bash run.sh train TransE wn18rr $1 fake40 512 256 500 6.0 0.5 0.00005 80000 8 --fake 40
bash run.sh train TransE wn18rr $1 CLF40_soft 512 256 500 6.0 0.5 0.00005 80000 8 --fake 40 --method clf -init ./models/TransE_wn18rr_fake40 --num 1
bash run.sh train TransE wn18rr $1 CLF40_hard 512 256 500 6.0 0.5 0.00005 80000 8 --fake 40 --method clf -init ./models/TransE_wn18rr_fake40 --num 1000

bash run.sh train TransE YAGO3-10 $1 baseline 512 256 100 12.0 0.5 0.001 100000 8
bash run.sh train TransE YAGO3-10 $1 baseline_soft 512 256 100 12.0 0.5 0.001 100000 8 --method clf -init ./models/TransE_YAGO3-10_baseline --num 1
bash run.sh train TransE YAGO3-10 $1 baseline_hard 512 256 100 12.0 0.5 0.001 100000 8 --method clf -init ./models/TransE_YAGO3-10_baseline --num 1000
bash run.sh train TransE YAGO3-10 $1 fake100 512 256 100 12.0 0.5 0.001 100000 8 --fake 100
bash run.sh train TransE YAGO3-10 $1 CLF100_soft 512 256 100 12.0 0.5 0.001 100000 8 --fake 100 --method clf -init ./models/TransE_YAGO3-10_fake100 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF100_hard 512 256 100 12.0 0.5 0.001 100000 8 --fake 100 --method clf -init ./models/TransE_YAGO3-10_fake100 --num 1000
bash run.sh train TransE YAGO3-10 $1 fake70 512 256 100 12.0 0.5 0.001 100000 8 --fake 70
bash run.sh train TransE YAGO3-10 $1 CLF70_soft 512 256 100 12.0 0.5 0.001 100000 8 --fake 70 --method clf -init ./models/TransE_YAGO3-10_fake70 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF70_hard 512 256 100 12.0 0.5 0.001 100000 8 --fake 70 --method clf -init ./models/TransE_YAGO3-10_fake70 --num 1000
bash run.sh train TransE YAGO3-10 $1 fake40 512 256 100 12.0 0.5 0.001 100000 8 --fake 40
bash run.sh train TransE YAGO3-10 $1 CLF40_soft 512 256 100 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1
bash run.sh train TransE YAGO3-10 $1 CLF40_hard 512 256 100 12.0 0.5 0.001 100000 8 --fake 40 --method clf -init ./models/TransE_YAGO3-10_fake40 --num 1000



bash run.sh test TransE YAGO3-10 1 CLF100_hard
bash run.sh test TransE YAGO3-10 1 CLF100_soft
bash run.sh test TransE YAGO3-10 1 CLF70_hard
bash run.sh test TransE YAGO3-10 1 CLF70_soft
bash run.sh test TransE YAGO3-10 1 CLF40_hard
bash run.sh test TransE YAGO3-10 1 CLF40_soft

bash run.sh test TransE wn18rr 1 CLF100_hard
bash run.sh test TransE wn18rr 1 CLF100_soft
bash run.sh test TransE wn18rr 1 CLF70_hard
bash run.sh test TransE wn18rr 1 CLF70_soft
bash run.sh test TransE wn18rr 1 CLF40_hard
bash run.sh test TransE wn18rr 1 CLF40_soft

bash run.sh test TransE FB15k-237 1 CLFPath100_hard
bash run.sh test TransE FB15k-237 1 CLFPath100_soft
bash run.sh test TransE FB15k-237 1 CLFPath70_hard
bash run.sh test TransE FB15k-237 1 CLFPath70_soft
bash run.sh test TransE FB15k-237 1 CLFPath40_hard
bash run.sh test TransE FB15k-237 1 CLFPath40_soft