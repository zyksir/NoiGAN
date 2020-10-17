 bash run.sh train TransE FB15k-237 $1 baseline 512 256 200 9.0 1.0 0.00005 100000 16
 bash run.sh train TransE wn18rr $1 baseline 512 256 200 6.0 0.5 0.00005 80000 8
 bash run.sh train TransE YAGO3-10 $1 baseline 512 256 50 12.0 0.5 0.001 100000 8


bash run.sh train TransE FB15k-237 $1 fake100 512 256 200 9.0 1.0 0.00005 100000 16 \
        --fake 100
bash run.sh train TransE wn18rr $1 fake100 512 256 200 6.0 0.5 0.00005 80000 8 \
        --fake 100
bash run.sh train TransE YAGO3-10 $1 fake100 512 256 50 12.0 0.5 0.001 100000 8 \
        --fake 100


bash run.sh train TransE FB15k-237 $1 LT100 512 256 200 9.0 1.0 0.00005 100000 16 \
        --fake 100 --method LT
bash run.sh train TransE wn18rr $1 LT100 512 256 200 6.0 0.5 0.00005 80000 8 \
        --fake 100 --method LT
bash run.sh train TransE YAGO3-10 $1 LT100 512 256 50 12.0 0.5 0.001 100000 8 \
        --fake 100 --method LT

bash run.sh train TransE FB15k-237 $1 NoiGAN_soft100 512 256 200 9.0 1.0 0.00005 100000 16 \
        --fake 100 --method NoiGAN -init ./models/TransE_FB15k-237_fake100

bash run.sh train TransE FB15k-237 7 CLF_soft100 512 256 200 9.0 1.0 0.00005 100000 16 \
        --fake 100 --method CLF -init ./models/TransE_FB15k-237_fake100
bash run.sh train TransE YAGO3-10 1 CLF_soft100 512 256 50 12.0 0.5 0.001 100000 8 \
        --fake 100 --method CLF -init ./models/TransE_YAGO3-10_fake100
