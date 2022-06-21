python train.py --experiment_name='test_experiment_original' --stage='chairs'                        --run_name='chairs' --gpu='0' --num_steps 50000  --batchSize 11 --testBatchSize=4 --lr=0.0004   --image_size 320 448  --wdecay 0.0001   --freeze_bn=0 --gamma=0.8  --workers=2 --no_4d_corr --num_corr_channels=4 --start_epoch=0  --thread=16 
python train.py --experiment_name='test_experiment_original' --stage='things' --weights 'chairs.pth' --run_name='things' --gpu='0' --num_steps 100000 --batchSize 4  --testBatchSize=4 --lr 0.000125 --image_size 448 768  --wdecay 0.0001   --freeze_bn=1 --gamma=0.8  --workers=2 --no_4d_corr --num_corr_channels=4 
python train.py --experiment_name='test_experiment_original' --stage='sintel' --weights 'things.pth' --run_name='sintel' --gpu='0' --num_steps 100000 --batchSize 4  --testBatchSize=4 --lr 0.000125 --image_size 384 832  --wdecay 0.00001  --freeze_bn=1 --gamma=0.85 --workers=2 --no_4d_corr --num_corr_channels=4 
python train.py --experiment_name='test_experiment_original' --stage='kitti'  --weights 'sintel.pth' --run_name='kitti'  --gpu='0' --num_steps 50000  --batchSize 4  --testBatchSize=4 --lr 0.0001   --image_size 320 1024 --wdecay 0.00001  --freeze_bn=1 --gamma=0.85 --workers=2 --no_4d_corr --num_corr_channels=4 