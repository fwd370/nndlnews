#!/bin/sh

nohup $(/
	python gru_generate.py --run 10 --epoch 10 --lr 0.005 --h-dim 32;\
	python gru_generate.py --run 11 --epoch 10 --lr 0.005 --h-dim 64;\
	python gru_generate.py --run 12 --epoch 10 --lr 0.005 --h-dim 128;\
	python gru_generate.py --run 13 --epoch 10 --lr 0.009 --h-dim 256;\
        python gru_generate.py --run 14 --epoch 10 --lr 0.009 --h-dim 512;\
        python gru_generate.py --run 15 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python gru_generate.py --run 16 --epoch 10 --lr 0.0005 --h-dim 32;\
        python gru_generate.py --run 17 --epoch 10 --lr 0.0005 --h-dim 64;\
        python gru_generate.py --run 18 --epoch 10 --lr 0.0005 --h-dim 128;\
        python gru_generate.py --run 19 --epoch 10 --lr 0.0009 --h-dim 256;\
        python gru_generate.py --run 20 --epoch 10 --lr 0.0009 --h-dim 512;\
        python gru_generate.py --run 21 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python gru_generate.py --run 22 --epoch 10 --lr 0.005 --enc-dim 32;\
        python gru_generate.py --run 23 --epoch 10 --lr 0.005 --enc-dim 64;\
        python gru_generate.py --run 24 --epoch 10 --lr 0.005 --enc-dim 128;\
        python gru_generate.py --run 25 --epoch 10 --lr 0.009 --enc-dim 256;\
        python gru_generate.py --run 26 --epoch 10 --lr 0.009 --enc-dim 512;\
        python gru_generate.py --run 27 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python gru_generate.py --run 28 --epoch 10 --lr 0.005 --dec-dim 32;\
        python gru_generate.py --run 29 --epoch 10 --lr 0.005 --dec-dim 64;\
        python gru_generate.py --run 30 --epoch 10 --lr 0.005 --dec-dim 128;\
        python gru_generate.py --run 31 --epoch 10 --lr 0.009 --dec-dim 256;\
        python gru_generate.py --run 32 --epoch 10 --lr 0.009 --dec-dim 512;\
        python gru_generate.py --run 33 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python gru_generate.py --run 34 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python gru_generate.py --run 35 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python gru_generate.py --run 36 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python gru_generate.py --run 37 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python gru_generate.py --run 38 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python gru_generate.py --run 39 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python gru_generate.py --run 40 --epoch 10 --lr 0.0005 --bs 2;\
        python gru_generate.py --run 41 --epoch 10 --lr 0.0005 --bs 8;\
        python gru_generate.py --run 42 --epoch 10 --lr 0.0005 --bs 16;\
        python gru_generate.py --run 43 --epoch 10 --lr 0.0009 --bs 32;\
        python gru_generate.py --run 44 --epoch 10 --lr 0.0009 --bs 64;\
        python gru_generate.py --run 45 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python gru_generate.py --run 46 --epoch 10 --lr 0.005 --clip 5.0;\
        python gru_generate.py --run 47 --epoch 10 --lr 0.005 --clip 4.0;\
        python gru_generate.py --run 48 --epoch 10 --lr 0.005 --clip 3.0;\
        python gru_generate.py --run 49 --epoch 10 --lr 0.009 --clip 6.0;\
        python gru_generate.py --run 50 --epoch 10 --lr 0.009 --clip 7.0;\
        python gru_generate.py --run 51 --epoch 10 --lr 0.009 --clip 10.0\
) &
