bash run.sh train RotatE wn18rr $1 baseline 512 1024 250 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE wn18rr $1 fake100 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 100
bash run.sh train RotatE wn18rr $1 fake70 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 70
bash run.sh train RotatE wn18rr $1 fake50 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 50
bash run.sh train RotatE wn18rr $1 fake30 512 1024 250 6.0 0.5 0.00005 80000 8 -de --fake 30

bash run.sh train RotatE FB15k-237 $1 baseline 1024 256 250 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE FB15k-237 $1 fake100 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 100
bash run.sh train RotatE FB15k-237 $1 fake70 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 70
bash run.sh train RotatE FB15k-237 $1 fake50 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 50
bash run.sh train RotatE FB15k-237 $1 fake30 1024 256 250 9.0 1.0 0.00005 100000 16 -de --fake 30

bash run.sh train RotatE YAGO3-10 $1 baseline 1024 400 50 24.0 1.0 0.0002 100000 4 -de
bash run.sh train RotatE YAGO3-10 $1 fake100 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 100
bash run.sh train RotatE YAGO3-10 $1 fake70 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 70
bash run.sh train RotatE YAGO3-10 $1 fake50 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 50
bash run.sh train RotatE YAGO3-10 $1 fake30 1024 400 50 24.0 1.0 0.0002 100000 4 -de --fake 30