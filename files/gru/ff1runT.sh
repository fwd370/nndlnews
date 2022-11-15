#!/bin/sh

nohup $(/
	python gru_generate_full.py --run 210 --epoch 10 --lr 0.005 --h-dim 32;\
	python gru_generate_full.py --run 211 --epoch 10 --lr 0.005 --h-dim 64;\
	python gru_generate_full.py --run 212 --epoch 10 --lr 0.005 --h-dim 128;\
	python gru_generate_full.py --run 213 --epoch 10 --lr 0.009 --h-dim 256;\
        python gru_generate_full.py --run 214 --epoch 10 --lr 0.009 --h-dim 512;\
        python gru_generate_full.py --run 215 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python gru_generate_full.py --run 216 --epoch 10 --lr 0.0005 --h-dim 32;\
        python gru_generate_full.py --run 217 --epoch 10 --lr 0.0005 --h-dim 64;\
        python gru_generate_full.py --run 218 --epoch 10 --lr 0.0005 --h-dim 128;\
        python gru_generate_full.py --run 219 --epoch 10 --lr 0.0009 --h-dim 256;\
        python gru_generate_full.py --run 220 --epoch 10 --lr 0.0009 --h-dim 512;\
        python gru_generate_full.py --run 221 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python gru_generate_full.py --run 222 --epoch 10 --lr 0.005 --enc-dim 32;\
        python gru_generate_full.py --run 223 --epoch 10 --lr 0.005 --enc-dim 64;\
        python gru_generate_full.py --run 224 --epoch 10 --lr 0.005 --enc-dim 128;\
        python gru_generate_full.py --run 225 --epoch 10 --lr 0.009 --enc-dim 256;\
        python gru_generate_full.py --run 226 --epoch 10 --lr 0.009 --enc-dim 512;\
        python gru_generate_full.py --run 227 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python gru_generate_full.py --run 228 --epoch 10 --lr 0.005 --dec-dim 32;\
        python gru_generate_full.py --run 229 --epoch 10 --lr 0.005 --dec-dim 64;\
        python gru_generate_full.py --run 230 --epoch 10 --lr 0.005 --dec-dim 128;\
        python gru_generate_full.py --run 231 --epoch 10 --lr 0.009 --dec-dim 256;\
        python gru_generate_full.py --run 232 --epoch 10 --lr 0.009 --dec-dim 512;\
        python gru_generate_full.py --run 233 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python gru_generate_full.py --run 234 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python gru_generate_full.py --run 235 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python gru_generate_full.py --run 236 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python gru_generate_full.py --run 237 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python gru_generate_full.py --run 238 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python gru_generate_full.py --run 239 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python gru_generate_full.py --run 240 --epoch 10 --lr 0.0005 --bs 2;\
        python gru_generate_full.py --run 241 --epoch 10 --lr 0.0005 --bs 8;\
        python gru_generate_full.py --run 242 --epoch 10 --lr 0.0005 --bs 16;\
        python gru_generate_full.py --run 243 --epoch 10 --lr 0.0009 --bs 32;\
        python gru_generate_full.py --run 244 --epoch 10 --lr 0.0009 --bs 64;\
        python gru_generate_full.py --run 245 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python gru_generate_full.py --run 246 --epoch 10 --lr 0.005 --clip 5.0;\
        python gru_generate_full.py --run 247 --epoch 10 --lr 0.005 --clip 4.0;\
        python gru_generate_full.py --run 248 --epoch 10 --lr 0.005 --clip 3.0;\
        python gru_generate_full.py --run 249 --epoch 10 --lr 0.009 --clip 6.0;\
        python gru_generate_full.py --run 250 --epoch 10 --lr 0.009 --clip 7.0;\
        python gru_generate_full.py --run 251 --epoch 10 --lr 0.009 --clip 10.0\
) &
