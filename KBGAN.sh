bash run.sh train TransE FB15k-237 $1 KBGAN 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_baseline --gen_init ./models/DistMult_FB15k-237_baseline --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN 512 256 500 6.0 0.5 0.00005 80000 8 --method KBGAN -init ./models/TransE_wn18rr_baseline --gen_init ./models/DistMult_wn18rr_baseline --gen_dim 250
bash run.sh train TransE YAGO3-10 $1 KBGAN 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_baseline --gen_init ./models/DistMult_YAGO3-10_baseline --gen_dim 50

bash run.sh train TransE FB15k-237 $1 KBGAN_fake100 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_fake100 --gen_init ./models/DistMult_FB15k-237_fake100 --fake 10 --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN_fake100 512 256 500 6.0 0.5 0.00005 80000 8 --method KBGAN -init ./models/TransE_wn18rr_fake100 --gen_init ./models/DistMult_fake100_baseline --fake 100 --gen_dim 250
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake100 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake100 --gen_init ./models/DistMult_YAGO3-10_fake100 --fake 10 --gen_dim 50

bash run.sh train TransE FB15k-237 $1 KBGAN_fake70 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_fake70 --gen_init ./models/DistMult_FB15k-237_fake70 --fake 70 --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN_fake70 512 256 500 6.0 0.5 0.00005 80000 8 --method KBGAN -init ./models/TransE_wn18rr_fake70 --gen_init ./models/DistMult_fake100_70 --fake 70 --gen_dim 250
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake70 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake70 --gen_init ./models/DistMult_YAGO3-10_fake70 --fake 70 --gen_dim 50

bash run.sh train TransE FB15k-237 $1 KBGAN_fake50 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_fake50 --gen_init ./models/DistMult_FB15k-237_fake50 --fake 50 --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN_fake50 512 256 500 6.0 0.5 0.00005 80000 8 --method KBGAN -init ./models/TransE_wn18rr_fake50 --gen_init ./models/DistMult_fake100_50 --fake 50 --gen_dim 250
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake50 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake50 --gen_init ./models/DistMult_YAGO3-10_fake50 --fake 50 --gen_dim 50

bash run.sh train TransE FB15k-237 $1 KBGAN_fake30 1024 256 1000 9.0 1.0 0.00005 100000 16 --method KBGAN -init ./models/TransE_FB15k-237_fake30 --gen_init ./models/DistMult_FB15k-237_fake30 --fake 30 --gen_dim 250
bash run.sh train TransE wn18rr $1 KBGAN_fake30 512 256 500 6.0 0.5 0.00005 80000 8 --method KBGAN -init ./models/TransE_wn18rr_fake30 --gen_init ./models/DistMult_fake100_30 --fake 30 --gen_dim 250
bash run.sh train TransE YAGO3-10 $1 KBGAN_fake30 512 1024 250 12.0 0.5 0.001 100000 8 --method KBGAN -init ./models/TransE_YAGO3-10_fake30 --gen_init ./models/DistMult_YAGO3-10_fake30 --fake 30 --gen_dim 50
