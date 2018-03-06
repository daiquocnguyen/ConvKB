CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 0 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 1 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 2 &
CUDA_VISIBLE_DEVICES=0 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 3 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 4 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 5 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 6 &
CUDA_VISIBLE_DEVICES=1 nohup python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 7 &


