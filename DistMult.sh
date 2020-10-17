# bash run.sh train DistMult FB15k-237 $1 baseline 1024 256 200 200.0 1.0 0.001 100000 16 -r 0.00001
# bash run.sh train DistMult wn18rr $1 baseline 512 1024 200 200.0 1.0 0.002 80000 8 -r 0.000005
# bash run.sh train DistMult YAGO3-10 $1 baseline 1024 400 50 24.0 1.0 0.0002 100000 4 -r 0.00001


bash run.sh train DistMult FB15k-237 $1 fake100 1024 256 200 200.0 1.0 0.001 100000 16 -r 0.00001 \
        --fake 100
bash run.sh train DistMult wn18rr $1 fake100 512 1024 200 200.0 1.0 0.002 80000 8 -r 0.000005 \
        --fake 100
bash run.sh train DistMult YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -r 0.00001 \
        --fake 100


bash run.sh train DistMult FB15k-237 $1 LT100 1024 256 200 200.0 1.0 0.001 100000 16 -r 0.00001 \
        --fake 100 --method LT
bash run.sh train DistMult wn18rr $1 LT100 512 1024 200 200.0 1.0 0.002 80000 8 -r 0.000005 \
        --fake 100 --method LT
bash run.sh train DistMult YAGO3-10 $1 LT100 1024 400 50 24.0 1.0 0.0002 100000 4 -r 0.00001 \
        --fake 100 --method LT



