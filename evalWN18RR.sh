CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 0 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 2 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 3 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 4 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 5 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 6 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 7 &