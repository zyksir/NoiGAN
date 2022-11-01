for i in 50 100 250 500 750 1000
do
  bash run.sh train RotatE FB15k $1 dim$i_fake30 1024 256 $i 24.0 1.0 0.0001 100000 16 -de --fake 30 --no_save
  bash run.sh train RotatE FB15k-237 $1 dim$i_fake30 1024 256 $i 9.0 1.0 0.00005 100000 16 -de --fake 30 --no_save
  bash run.sh train RotatE wn18 $1 dim$i_fake30 512 1024 $i 12.0 0.5 0.0001 80000 8 -de --fake 30 --no_save
  bash run.sh train RotatE wn18rr $1 dim$i_fake30 512 1024 $i 6.0 0.5 0.00005 80000 8 -de --fake 30 --no_save
done


for i in 50 100 250 500 750 1000
do
  bash run.sh train TransE FB15k $1 dim$i_fake30 1024 256 $i 24.0 1.0 0.0001 150000 16 --fake 30 --no_save
  bash run.sh train TransE FB15k-237 $1 dim$i_fake30 1024 256 $i 9.0 1.0 0.00005 100000 16 --fake 30 --no_save
  bash run.sh train TransE wn18 $1 dim$i_fake30 512 1024 $i 12.0 0.5 0.0001 80000 8 --fake 30 --no_save
  bash run.sh train TransE wn18rr $1 dim$i_fake30 512 1024 $i 6.0 0.5 0.00005 80000 8 --fake 30 --no_save
done

for i in 50 100 250 500 750 1000
do
  bash run.sh train ComplEx FB15k $1 dim$i_fake30 1024 256 $i 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002 --fake 30 --no_save
  bash run.sh train ComplEx FB15k-237 $1 dim$i_fake30 1024 256 $i 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 --fake 30 --no_save
  bash run.sh train ComplEx wn18 $1 dim$i_fake30 512 1024 $i 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001 --fake 30 --no_save
  bash run.sh train ComplEx wn18rr $1 dim$i_fake30 512 1024 $i 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 --fake 30 --no_save
done

for i in 50 100 250 500 750 1000
do
  bash run.sh train DistMult FB15k $1 dim$i_fake30 1024 256 $i 500.0 1.0 0.001 150000 16 -r 0.000002 --fake 30 --no_save
  bash run.sh train DistMult FB15k-237 $1 dim$i_fake30 1024 256 $i 200.0 1.0 0.001 100000 16 -r 0.00001 --fake 30 --no_save
  bash run.sh train DistMult wn18 $1 dim$i_fake30 512 1024 $i 200.0 1.0 0.001 80000 8 -r 0.00001 --fake 30 --no_save
  bash run.sh train DistMult wn18rr $1 dim$i_fake30 512 1024 $i 200.0 1.0 0.002 80000 8 -r 0.000005 --fake 30 --no_save
done

for i in 50 100 200 300 400 500
do
  bash run.sh train RotatE YAGO3-10 $1 dim$i_fake30 1024 400 $i 24.0 1.0 0.0002 100000 4 -de --fake 30 --no_save
  bash run.sh train TransE YAGO3-10 $1 dim$i_fake30 1024 400 $i 24.0 1.0 0.0002 100000 4 -de --fake 30 --no_save
  bash run.sh train ComplEx YAGO3-10 $1 dim$i_fake30 1024 400 $i 24.0 1.0 0.0002 100000 4 -de --fake 30 --no_save
  bash run.sh train DistMult YAGO3-10 $1 dim$i_fake30 1024 400 $i 24.0 1.0 0.0002 100000 4 -de --fake 30 --no_save
done