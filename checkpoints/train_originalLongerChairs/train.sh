# exact same arguments as original implementation, except path related
# python train.py --experiment_name='train_originalLongerChairs' --stage='chairs'                                                                         --run_name='chairs' --gpu='0,1,2,3' --num_steps 100000 --batchSize 12 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8   --start_epoch=0  --thread=16 
# python train.py --experiment_name='train_originalLongerChairs' --stage='things' --weights 'checkpoints/train_originalLongerChairs/models/chairs.pth'    --run_name='things' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 448 768  --wdecay 0.0001   --freeze_bn=1 --gamma=0.8     

# Only C+T for now
python train.py --experiment_name='train_originalLongerChairs' --stage='sintel' --weights 'checkpoints/train_originalLongerChairs/models/things.pth'    --run_name='sintel' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 384 832  --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    
python train.py --experiment_name='train_originalLongerChairs' --stage='kitti'  --weights 'checkpoints/train_originalLongerChairs/models/sintel.pth'    --run_name='kitti'  --gpu='0,1,2,3' --num_steps 50000  --batchSize 8  --testBatchSize=4 --lr 0.0001   --image_size 320 1024 --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    
