# exact same arguments as original implementation, except path related
# python train.py --experiment_name='train_originalGMA' --stage='chairs'                                                              --run_name='chairs' --gpu='0,1,2,3' --num_steps 100000 --batchSize 12 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8 --use_gma   --start_epoch=0  --thread=16 
# python train.py --experiment_name='train_originalGMA' --stage='things' --weights 'checkpoints/train_originalGMA/models/chairs.pth'  --run_name='things' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 448 768  --wdecay 0.0001   --freeze_bn=1 --gamma=0.8 --use_gma     

# Only C+T for now
python train.py --experiment_name='train_originalGMA' --stage='sintel' --weights 'checkpoints/train_originalGMA/models/things.pth'  --run_name='sintel' --gpu='0,1,2,3' --num_steps 100000 --batchSize 8  --testBatchSize=4 --lr 0.000125 --image_size 384 832  --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    --use_gma
python train.py --experiment_name='train_originalGMA' --stage='kitti'  --weights 'checkpoints/train_originalGMA/models/sintel.pth'  --run_name='kitti'  --gpu='0,1,2,3' --num_steps 50000  --batchSize 8  --testBatchSize=4 --lr 0.0001   --image_size 320 1024 --wdecay 0.00001  --freeze_bn=1 --gamma=0.85    --use_gma
