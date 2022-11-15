#!/bin/sh

nohup $(/
	python rnn_generate_full.py --run 1210 --epoch 10 --lr 0.005 --h-dim 32;\
	python rnn_generate_full.py --run 1211 --epoch 10 --lr 0.005 --h-dim 64;\
	python rnn_generate_full.py --run 1212 --epoch 10 --lr 0.005 --h-dim 128;\
	python rnn_generate_full.py --run 1213 --epoch 10 --lr 0.009 --h-dim 256;\
        python rnn_generate_full.py --run 1214 --epoch 10 --lr 0.009 --h-dim 512;\
        python rnn_generate_full.py --run 1215 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python rnn_generate_full.py --run 1216 --epoch 10 --lr 0.0005 --h-dim 32;\
        python rnn_generate_full.py --run 1217 --epoch 10 --lr 0.0005 --h-dim 64;\
        python rnn_generate_full.py --run 1218 --epoch 10 --lr 0.0005 --h-dim 128;\
        python rnn_generate_full.py --run 1219 --epoch 10 --lr 0.0009 --h-dim 256;\
        python rnn_generate_full.py --run 1220 --epoch 10 --lr 0.0009 --h-dim 512;\
        python rnn_generate_full.py --run 1221 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python rnn_generate_full.py --run 1222 --epoch 10 --lr 0.005 --enc-dim 32;\
        python rnn_generate_full.py --run 1223 --epoch 10 --lr 0.005 --enc-dim 64;\
        python rnn_generate_full.py --run 1224 --epoch 10 --lr 0.005 --enc-dim 128;\
        python rnn_generate_full.py --run 1225 --epoch 10 --lr 0.009 --enc-dim 256;\
        python rnn_generate_full.py --run 1226 --epoch 10 --lr 0.009 --enc-dim 512;\
        python rnn_generate_full.py --run 1227 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python rnn_generate_full.py --run 1228 --epoch 10 --lr 0.005 --dec-dim 32;\
        python rnn_generate_full.py --run 1229 --epoch 10 --lr 0.005 --dec-dim 64;\
        python rnn_generate_full.py --run 1230 --epoch 10 --lr 0.005 --dec-dim 128;\
        python rnn_generate_full.py --run 1231 --epoch 10 --lr 0.009 --dec-dim 256;\
        python rnn_generate_full.py --run 1232 --epoch 10 --lr 0.009 --dec-dim 512;\
        python rnn_generate_full.py --run 1233 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python rnn_generate_full.py --run 1234 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python rnn_generate_full.py --run 1235 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python rnn_generate_full.py --run 1236 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python rnn_generate_full.py --run 1237 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python rnn_generate_full.py --run 1238 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python rnn_generate_full.py --run 1239 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python rnn_generate_full.py --run 1240 --epoch 10 --lr 0.0005 --bs 2;\
        python rnn_generate_full.py --run 1241 --epoch 10 --lr 0.0005 --bs 8;\
        python rnn_generate_full.py --run 1242 --epoch 10 --lr 0.0005 --bs 16;\
        python rnn_generate_full.py --run 1243 --epoch 10 --lr 0.0009 --bs 32;\
        python rnn_generate_full.py --run 1244 --epoch 10 --lr 0.0009 --bs 64;\
        python rnn_generate_full.py --run 1245 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python rnn_generate_full.py --run 1246 --epoch 10 --lr 0.005 --clip 5.0;\
        python rnn_generate_full.py --run 1247 --epoch 10 --lr 0.005 --clip 4.0;\
        python rnn_generate_full.py --run 1248 --epoch 10 --lr 0.005 --clip 3.0;\
        python rnn_generate_full.py --run 1249 --epoch 10 --lr 0.009 --clip 6.0;\
        python rnn_generate_full.py --run 1250 --epoch 10 --lr 0.009 --clip 7.0;\
        python rnn_generate_full.py --run 1251 --epoch 10 --lr 0.009 --clip 10.0\
) &
