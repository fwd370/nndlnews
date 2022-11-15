#!/bin/sh

nohup $(/
	python rnn_generate.py --run 1010 --epoch 10 --lr 0.005 --h-dim 32;\
	python rnn_generate.py --run 1011 --epoch 10 --lr 0.005 --h-dim 64;\
	python rnn_generate.py --run 1012 --epoch 10 --lr 0.005 --h-dim 128;\
	python rnn_generate.py --run 1013 --epoch 10 --lr 0.009 --h-dim 256;\
        python rnn_generate.py --run 1014 --epoch 10 --lr 0.009 --h-dim 512;\
        python rnn_generate.py --run 1015 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python rnn_generate.py --run 1016 --epoch 10 --lr 0.0005 --h-dim 32;\
        python rnn_generate.py --run 1017 --epoch 10 --lr 0.0005 --h-dim 64;\
        python rnn_generate.py --run 1018 --epoch 10 --lr 0.0005 --h-dim 128;\
        python rnn_generate.py --run 1019 --epoch 10 --lr 0.0009 --h-dim 256;\
        python rnn_generate.py --run 1020 --epoch 10 --lr 0.0009 --h-dim 512;\
        python rnn_generate.py --run 1021 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python rnn_generate.py --run 1022 --epoch 10 --lr 0.005 --enc-dim 32;\
        python rnn_generate.py --run 1023 --epoch 10 --lr 0.005 --enc-dim 64;\
        python rnn_generate.py --run 1024 --epoch 10 --lr 0.005 --enc-dim 128;\
        python rnn_generate.py --run 1025 --epoch 10 --lr 0.009 --enc-dim 256;\
        python rnn_generate.py --run 1026 --epoch 10 --lr 0.009 --enc-dim 512;\
        python rnn_generate.py --run 1027 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python rnn_generate.py --run 1028 --epoch 10 --lr 0.005 --dec-dim 32;\
        python rnn_generate.py --run 1029 --epoch 10 --lr 0.005 --dec-dim 64;\
        python rnn_generate.py --run 1030 --epoch 10 --lr 0.005 --dec-dim 128;\
        python rnn_generate.py --run 1031 --epoch 10 --lr 0.009 --dec-dim 256;\
        python rnn_generate.py --run 1032 --epoch 10 --lr 0.009 --dec-dim 512;\
        python rnn_generate.py --run 1033 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python rnn_generate.py --run 1034 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python rnn_generate.py --run 1035 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python rnn_generate.py --run 1036 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python rnn_generate.py --run 1037 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python rnn_generate.py --run 1038 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python rnn_generate.py --run 1039 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python rnn_generate.py --run 1040 --epoch 10 --lr 0.0005 --bs 2;\
        python rnn_generate.py --run 1041 --epoch 10 --lr 0.0005 --bs 8;\
        python rnn_generate.py --run 1042 --epoch 10 --lr 0.0005 --bs 16;\
        python rnn_generate.py --run 1043 --epoch 10 --lr 0.0009 --bs 32;\
        python rnn_generate.py --run 1044 --epoch 10 --lr 0.0009 --bs 64;\
        python rnn_generate.py --run 1045 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python rnn_generate.py --run 1046 --epoch 10 --lr 0.005 --clip 5.0;\
        python rnn_generate.py --run 1047 --epoch 10 --lr 0.005 --clip 4.0;\
        python rnn_generate.py --run 1048 --epoch 10 --lr 0.005 --clip 3.0;\
        python rnn_generate.py --run 1049 --epoch 10 --lr 0.009 --clip 6.0;\
        python rnn_generate.py --run 1050 --epoch 10 --lr 0.009 --clip 7.0;\
        python rnn_generate.py --run 1051 --epoch 10 --lr 0.009 --clip 10.0\
) &
