bash run.sh train RotatE wn18rr $1 baseline 512 1024 250 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE wn18rr $1 baseline_soft 512 1024 250 6.0 0.5 0.00005 80000 8 -de --method clf -init ./models/RotatE_wn18rr_baseline  --num 1
bash run.sh train RotatE wn18rr $1 baseline_hard 512 1024 250 6.0 0.5 0.00005 80000 8 -de --method clf -init ./models/RotatE_wn18rr_baseline  --num 1000
bash run.sh train RotatE wn18rr $1 fake100_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100
bash run.sh train RotatE wn18rr $1 CLF100_soft_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100 --method clf -init ./models/RotatE_wn18rr_fake100_new --num 1
bash run.sh train RotatE wn18rr $1 CLF100_hard_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100 --method clf -init ./models/RotatE_wn18rr_fake100_new --num 1000
bash run.sh train RotatE wn18rr $1 fake70_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70
bash run.sh train RotatE wn18rr $1 CLF70_soft_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70 --method clf -init ./models/RotatE_wn18rr_fake70_new --num 1
bash run.sh train RotatE wn18rr $1 CLF70_hard_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70 --method clf -init ./models/RotatE_wn18rr_fake70_new --num 1000
bash run.sh train RotatE wn18rr $1 fake40_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 40
bash run.sh train RotatE wn18rr $1 CLF40_soft_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 40 --method clf -init ./models/RotatE_wn18rr_fake40_new --num 1
bash run.sh train RotatE wn18rr $1 CLF40_hard_new 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 40 --method clf -init ./models/RotatE_wn18rr_fake40_new --num 1000

bash run.sh train RotatE FB15k-237 $1 baseline 1024 256 250 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE FB15k-237 $1 fakePath100 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100
bash run.sh train RotatE FB15k-237 $1 CLFPath100_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100 --method clf -init ./models/RotatE_FB15k-237_fakePath100  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath100_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path100 --method clf -init ./models/RotatE_FB15k-237_fakePath100  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath70 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70
bash run.sh train RotatE FB15k-237 $1 CLFPath70_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70 --method clf -init ./models/RotatE_FB15k-237_fakePath70  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath70_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path70 --method clf -init ./models/RotatE_FB15k-237_fakePath70  --num 1000
bash run.sh train RotatE FB15k-237 $1 fakePath40 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path40
bash run.sh train RotatE FB15k-237 $1 CLFPath40_soft 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path40 --method clf -init ./models/RotatE_FB15k-237_fakePath40  --num 1
bash run.sh train RotatE FB15k-237 $1 CLFPath40_hard 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake Path40 --method clf -init ./models/RotatE_FB15k-237_fakePath40  --num 1000

bash run.sh train RotatE YAGO3-10 $1 baseline_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de
bash run.sh train RotatE YAGO3-10 $1 baseline_soft_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --method clf -init ./models/RotatE_YAGO3-10_baseline --num 1
bash run.sh train RotatE YAGO3-10 $1 baseline_hard_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --method clf -init ./models/RotatE_YAGO3-10_baseline  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fake100_new 1024 400 100 24.0 1.0 0.001 100000 4 -de --fake 100 --warm_up_steps 80000
bash run.sh train RotatE YAGO3-10 $1 CLF100_soft_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 100 --method clf -init ./models/RotatE_YAGO3-10_fake100_new  --num 1
bash run.sh train RotatE YAGO3-10 $1 CLF100_hard_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 100 --method clf -init ./models/RotatE_YAGO3-10_fake100_new  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fake70_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 70
bash run.sh train RotatE YAGO3-10 $1 CLF70_soft_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 70 --method clf -init ./models/RotatE_YAGO3-10_fake70_new  --num 1
bash run.sh train RotatE YAGO3-10 $1 CLF70_hard_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 70 --method clf -init ./models/RotatE_YAGO3-10_fake70_new  --num 1000
bash run.sh train RotatE YAGO3-10 $1 fake40 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 40 --warm_up_steps 80000
bash run.sh train RotatE YAGO3-10 $1 CLF40_soft_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 40 --method clf -init ./models/RotatE_YAGO3-10_fake40_new  --num 1
bash run.sh train RotatE YAGO3-10 $1 CLF40_hard_new 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 40 --method clf -init ./models/RotatE_YAGO3-10_fake40_new  --num 1000

bash run.sh test RotatE wn18rr 1 fake100

bash run.sh test RotatE YAGO3-10 1 CLF100_hard
bash run.sh test RotatE YAGO3-10 1 CLF100_soft
bash run.sh test RotatE YAGO3-10 1 CLF70_hard
bash run.sh test RotatE YAGO3-10 1 CLF70_soft
bash run.sh test RotatE YAGO3-10 1 CLF40_hard
bash run.sh test RotatE YAGO3-10 1 CLF40_soft

bash run.sh test RotatE wn18rr 1 CLFPath100_hard
bash run.sh test RotatE wn18rr 1 CLF100_soft
bash run.sh test RotatE wn18rr 1 CLF70_hard
bash run.sh test RotatE wn18rr 1 CLF70_soft
bash run.sh test RotatE wn18rr 1 CLF40_hard
bash run.sh test RotatE wn18rr 1 CLF40_soft

bash run.sh test RotatE FB15k-237 1 CLFPath100_hard
bash run.sh test RotatE FB15k-237 1 CLFPath100_soft
bash run.sh test RotatE FB15k-237 1 CLFPath70_hard
bash run.sh test RotatE FB15k-237 1 CLFPath70_soft
bash run.sh test RotatE FB15k-237 1 CLFPath40_hard
bash run.sh test RotatE FB15k-237 1 CLFPath40_soft

bash run.sh test RotatE YAGO3-10 4 fake100
bash run.sh test RotatE YAGO3-10 4 fake70
bash run.sh test RotatE YAGO3-10 4 fake40
bash run.sh test RotatE FB15k-237 5 fakePath100
bash run.sh test RotatE FB15k-237 5 fakePath70
bash run.sh test RotatE FB15k-237 5 fakePath40

bash run.sh train RotatE YAGO3-10 $1 fake70 1024 400 100 24.0 1.0 0.0005 100000 4 -de --fake 70
bash run.sh test RotatE YAGO3-10 $1 fake70
bash run.sh train RotatE YAGO3-10 $1 fake100 1024 400 100 24.0 1.0 0.001 100000 4 -de --fake 100
bash run.sh test RotatE YAGO3-10 $1 fake100