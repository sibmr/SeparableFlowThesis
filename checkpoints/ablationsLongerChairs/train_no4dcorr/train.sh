python train.py --experiment_name='ablationsLongerChairs/train_no4dcorr' --stage='chairs'                                                                                   --run_name='chairs' --gpu='0,1,2,3' --num_steps 100000 --batchSize 12 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8  --workers=2 --no_4d_corr --start_epoch=0  --thread=16 
python train.py --experiment_name='ablationsLongerChairs/train_no4dcorr' --stage='things' --weights 'checkpoints/ablationsLongerChairs/train_no4dcorr/models/chairs.pth'    --run_name='things' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 448 768  --wdecay 0.0001   --freeze_bn=1 --gamma=0.8  --workers=2 --no_4d_corr 

# only C+T for now
# python train.py --experiment_name='ablationsLongerChairs/train_no4dcorr' --stage='sintel' --weights 'checkpoints/ablationsLongerChairs/train_no4dcorr/models/things.pth' --run_name='sintel' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 384 832  --wdecay 0.00001  --freeze_bn=1 --gamma=0.85 --workers=2 --no_4d_corr 
# python train.py --experiment_name='ablationsLongerChairs/train_no4dcorr' --stage='kitti'  --weights 'checkpoints/ablationsLongerChairs/train_no4dcorr/models/sintel.pth' --run_name='kitti'  --gpu='0,1,2,3' --num_steps 50000  --batchSize 8  --testBatchSize=4 --lr 0.0001   --image_size 320 1024 --wdecay 0.00001  --freeze_bn=1 --gamma=0.85 --workers=2 --no_4d_corr 