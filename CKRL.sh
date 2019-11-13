bash run.sh train TransE wn18rr $1 fake100 512 256 500 6.0 0.5 0.00005 80000 8 --fake 100 --method LT
bash run.sh train TransE wn18rr $1 fake70 512 256 500 6.0 0.5 0.00005 80000 8 --fake 70 --method LT
bash run.sh train TransE wn18rr $1 fake40 512 256 500 6.0 0.5 0.00005 80000 8 --fake 40 --method LT

bash run.sh train TransE YAGO3-10 $1 fake100 512 256 100 12.0 0.5 0.001 100000 8 --fake 100 --method LT
bash run.sh train TransE YAGO3-10 $1 fake70 512 256 100 12.0 0.5 0.001 100000 8 --fake 70 --method LT
bash run.sh train TransE YAGO3-10 $1 fake40 512 256 100 12.0 0.5 0.001 100000 8 --fake 40 --method LT

bash run.sh train TransE FB15k-237 $1 fakePath100 512 256 1000 9.0 1.0 0.00005 100000 16 --fake Path100 --method LT
bash run.sh train TransE FB15k-237 $1 fakePath70 512 256 1000 9.0 1.0 0.00005 100000 16 --fake Path70 --method LT
bash run.sh train TransE FB15k-237 $1 fakePath40 512 256 1000 9.0 1.0 0.00005 100000 16 --fake Path40 --method LT

