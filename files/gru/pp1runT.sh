#!/bin/sh

nohup $(/
	python gru_generate_preprocessed.py --run 110 --epoch 10 --lr 0.005 --h-dim 32;\
	python gru_generate_preprocessed.py --run 111 --epoch 10 --lr 0.005 --h-dim 64;\
	python gru_generate_preprocessed.py --run 112 --epoch 10 --lr 0.005 --h-dim 128;\
	python gru_generate_preprocessed.py --run 113 --epoch 10 --lr 0.009 --h-dim 256;\
        python gru_generate_preprocessed.py --run 114 --epoch 10 --lr 0.009 --h-dim 512;\
        python gru_generate_preprocessed.py --run 115 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python gru_generate_preprocessed.py --run 116 --epoch 10 --lr 0.0005 --h-dim 32;\
        python gru_generate_preprocessed.py --run 117 --epoch 10 --lr 0.0005 --h-dim 64;\
        python gru_generate_preprocessed.py --run 118 --epoch 10 --lr 0.0005 --h-dim 128;\
        python gru_generate_preprocessed.py --run 119 --epoch 10 --lr 0.0009 --h-dim 256;\
        python gru_generate_preprocessed.py --run 120 --epoch 10 --lr 0.0009 --h-dim 512;\
        python gru_generate_preprocessed.py --run 121 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python gru_generate_preprocessed.py --run 122 --epoch 10 --lr 0.005 --enc-dim 32;\
        python gru_generate_preprocessed.py --run 123 --epoch 10 --lr 0.005 --enc-dim 64;\
        python gru_generate_preprocessed.py --run 124 --epoch 10 --lr 0.005 --enc-dim 128;\
        python gru_generate_preprocessed.py --run 125 --epoch 10 --lr 0.009 --enc-dim 256;\
        python gru_generate_preprocessed.py --run 126 --epoch 10 --lr 0.009 --enc-dim 512;\
        python gru_generate_preprocessed.py --run 127 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python gru_generate_preprocessed.py --run 128 --epoch 10 --lr 0.005 --dec-dim 32;\
        python gru_generate_preprocessed.py --run 129 --epoch 10 --lr 0.005 --dec-dim 64;\
        python gru_generate_preprocessed.py --run 130 --epoch 10 --lr 0.005 --dec-dim 128;\
        python gru_generate_preprocessed.py --run 131 --epoch 10 --lr 0.009 --dec-dim 256;\
        python gru_generate_preprocessed.py --run 132 --epoch 10 --lr 0.009 --dec-dim 512;\
        python gru_generate_preprocessed.py --run 133 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python gru_generate_preprocessed.py --run 134 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python gru_generate_preprocessed.py --run 135 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python gru_generate_preprocessed.py --run 136 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python gru_generate_preprocessed.py --run 137 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python gru_generate_preprocessed.py --run 138 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python gru_generate_preprocessed.py --run 139 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python gru_generate_preprocessed.py --run 140 --epoch 10 --lr 0.0005 --bs 2;\
        python gru_generate_preprocessed.py --run 141 --epoch 10 --lr 0.0005 --bs 8;\
        python gru_generate_preprocessed.py --run 142 --epoch 10 --lr 0.0005 --bs 16;\
        python gru_generate_preprocessed.py --run 143 --epoch 10 --lr 0.0009 --bs 32;\
        python gru_generate_preprocessed.py --run 144 --epoch 10 --lr 0.0009 --bs 64;\
        python gru_generate_preprocessed.py --run 145 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python gru_generate_preprocessed.py --run 146 --epoch 10 --lr 0.005 --clip 5.0;\
        python gru_generate_preprocessed.py --run 147 --epoch 10 --lr 0.005 --clip 4.0;\
        python gru_generate_preprocessed.py --run 148 --epoch 10 --lr 0.005 --clip 3.0;\
        python gru_generate_preprocessed.py --run 149 --epoch 10 --lr 0.009 --clip 6.0;\
        python gru_generate_preprocessed.py --run 150 --epoch 10 --lr 0.009 --clip 7.0;\
        python gru_generate_preprocessed.py --run 151 --epoch 10 --lr 0.009 --clip 10.0\
) &
