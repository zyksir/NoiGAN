# bash run.sh train TransE NELL27k 1 base_clf 512 1024 500 12.0 0.5 0.0001 100000 8 --method clf -init ./models/TransE_NELL27k_baseline
#bash run.sh train TransE NELL27k 1 base_clf_hard 512 1024 500 12.0 0.5 0.0001 100000 8 --method clf -init ./models/TransE_NELL27k_baseline --num 1000
#bash run.sh train TransE YAGO3-10 1 base_clf_soft 512 1024 500 12.0 0.5 0.0001 100000 8 --method clf -init ./models/TransE_YAGO3-10_baseline --num 1
#bash run.sh train TransE YAGO3-10 1 base_clf_hard 512 1024 500 12.0 0.5 0.0001 100000 8 --method clf -init ./models/TransE_YAGO3-10_baseline --num 1000
bash run.sh train TransE YAGO3-10 1 fake10 512 1024 500 12.0 0.5 0.0001 100000 8 --fake 10
bash run.sh train TransE NELL27k 1 fake10 512 1024 500 12.0 0.5 0.0001 100000 8 --fake 10