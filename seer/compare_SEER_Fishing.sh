echo $HOSTNAME
git checkout tmp_prop_merge3
git pull
BATCH_SIZE=128
PROPERTY=red
DATASET=Cifar10
for MULTIPLIER in {12.5,25,50,100,200,400}
do
python3 ../breaching/breaching_dsnr.py $MULTIPLIER $BATCH_SIZE
done
bash train_rel.sh $BATCH_SIZE --dataset $DATASET --prop $PROPERTY
bash test_rel.sh $BATCH_SIZE --dataset $DATASET --prop $PROPERTY --checkpoint "<CHECKPOINT_JUST_TRAINED>"
