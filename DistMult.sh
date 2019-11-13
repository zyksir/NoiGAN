bash run.sh train DistMult wn18rr $1 baseline 512 1024 250 200.0 1.0 0.002 80000 8 -r 0.000005
bash run.sh train DistMult wn18rr $1 fake100 512 1024 250 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 100
bash run.sh train DistMult wn18rr $1 fake70 512 1024 250 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 70
bash run.sh train DistMult wn18rr $1 fake50 512 1024 250 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 50
bash run.sh train DistMult wn18rr $1 fake50 512 1024 250 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 30

bash run.sh train DistMult FB15k-237 $1 baseline 1024 256 250 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult FB15k-237 $1 fake100 1024 256 250 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 100
bash run.sh train DistMult FB15k-237 $1 fake70 1024 256 250 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 70
bash run.sh train DistMult FB15k-237 $1 fake50 1024 256 250 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 50
bash run.sh train DistMult FB15k-237 $1 fake30 1024 256 250 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 30

bash run.sh train DistMult YAGO3-10 $1 baseline 1024 400 50 24.0 1.0 0.0002 100000 4 -de
bash run.sh train DistMult YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 100
bash run.sh train DistMult YAGO3-10 $1 fake70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 70
bash run.sh train DistMult YAGO3-10 $1 fake50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 50
bash run.sh train DistMult YAGO3-10 $1 fake30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 30