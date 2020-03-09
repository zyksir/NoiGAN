bash run.sh train RotatE wn18rr 2 tmp 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 40 --method clf -init ./models/RotatE_wn18rr_fake40  --num 1000

bash run.sh train TransE FB15k-237 $1 KBGAN 1024 256 1000 9.0 1.0 0.00005 50000 16 --method KBGAN -init ./models/TransE_FB15k-237_baseline --gen_init ./models/DistMult_FB15k-237_baseline --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN 512 256 500 6.0 0.5 0.00005 40000 8 --method KBGAN -init ./models/TransE_wn18rr_baseline --gen_init ./models/DistMult_wn18rr_baseline --gen_dim 500
bash run.sh train TransE YAGO3-10 $1 KBGAN 512 1024 250 12.0 0.5 0.001 50000 8 --method KBGAN -init ./models/TransE_YAGO3-10_baseline --gen_init ./models/DistMult_YAGO3-10_baseline --gen_dim 50

\bash run.sh train TransE YAGO3-10 $1 KBGAN_fake100 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake100 --gen_init ./models/DistMult_YAGO3-10_fake100 --fake 100 --gen_dim 50
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake70 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake70 --gen_init ./models/DistMult_YAGO3-10_fake70 --fake 70 --gen_dim 50
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake40 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake40 --gen_init ./models/DistMult_YAGO3-10_fake40 --fake 40 --gen_dim 50


bash run.sh train TransE FB15k-237 4 KBGAN_fake40 1024 256 1000 9.0 1.0 0.00005 50000 16 --method KBGAN -init ./models/TransE_FB15k-237_fakePath40 --gen_init ./models/DistMult_FB15k-237_fakePath40 --fake Path40 --gen_dim 250
bash run.sh train TransE FB15k-237 5 KBGAN_fake70 1024 256 1000 9.0 1.0 0.00005 50000 16 --method KBGAN -init ./models/TransE_FB15k-237_fakePath70 --gen_init ./models/DistMult_FB15k-237_fakePath70 --fake Path70 --gen_dim 250
bash run.sh train TransE FB15k-237 6 KBGAN_fake100 1024 256 1000 9.0 1.0 0.00005 50000 16 --method KBGAN -init ./models/TransE_FB15k-237_fakePath100 --gen_init ./models/DistMult_FB15k-237_fakePath100 --fake Path100 --gen_dim 250


bash run.sh train TransE wn18rr 1 KBGAN_fake100 512 256 500 6.0 0.5 0.00005 40000 8 --method KBGAN -init ./models/TransE_wn18rr_fake100 --gen_init ./models/DistMult_wn18rr_fake100 --fake 100 --gen_dim 250
bash run.sh train TransE wn18rr 2 KBGAN_fake70 512 256 500 6.0 0.5 0.00005 40000 8 --method KBGAN -init ./models/TransE_wn18rr_fake70 --gen_init ./models/DistMult_wn18rr_fake70 --fake 70 --gen_dim 250
bash run.sh train TransE wn18rr 3 KBGAN_fake40 512 256 500 6.0 0.5 0.00005 40000 8 --method KBGAN -init ./models/TransE_wn18rr_fake40 --gen_init ./models/DistMult_wn18rr_fake40 --fake 40 --gen_dim 250
