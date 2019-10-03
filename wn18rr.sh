bash run.sh train TransE wn18rr 4 baseline 512 1024 500 6.0 0.5 0.00005 80000 8

bash run.sh train TransE wn18rr 4 fake10 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10
bash run.sh train TransE wn18rr 4 fake20 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20
bash run.sh train TransE wn18rr 4 fake40 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40

bash run.sh train TransE wn18rr 4 fake10 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10 --method LT
bash run.sh train TransE wn18rr 4 fake20 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20 --method LT
bash run.sh train TransE wn18rr 4 fake40 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40 --method LT

bash run.sh train TransE wn18rr 7 fake10 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 10 --method clf -init ./models/TransE_wn18rr_fake10
bash run.sh train TransE wn18rr 7 fake20 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 20 --method clf -init ./models/TransE_wn18rr_fake20
bash run.sh train TransE wn18rr 7 fake40 512 1024 500 6.0 0.5 0.00005 80000 8 --fake 40 --method clf -init ./models/TransE_wn18rr_fake40

