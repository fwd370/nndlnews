#!/bin/sh

nohup $(/
	python rnn_generate_preprocessed.py --run 1110 --epoch 10 --lr 0.005 --h-dim 32;\
	python rnn_generate_preprocessed.py --run 1111 --epoch 10 --lr 0.005 --h-dim 64;\
	python rnn_generate_preprocessed.py --run 1112 --epoch 10 --lr 0.005 --h-dim 128;\
	python rnn_generate_preprocessed.py --run 1113 --epoch 10 --lr 0.009 --h-dim 256;\
        python rnn_generate_preprocessed.py --run 1114 --epoch 10 --lr 0.009 --h-dim 512;\
        python rnn_generate_preprocessed.py --run 1115 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python rnn_generate_preprocessed.py --run 1116 --epoch 10 --lr 0.0005 --h-dim 32;\
        python rnn_generate_preprocessed.py --run 1117 --epoch 10 --lr 0.0005 --h-dim 64;\
        python rnn_generate_preprocessed.py --run 1118 --epoch 10 --lr 0.0005 --h-dim 128;\
        python rnn_generate_preprocessed.py --run 1119 --epoch 10 --lr 0.0009 --h-dim 256;\
        python rnn_generate_preprocessed.py --run 1120 --epoch 10 --lr 0.0009 --h-dim 512;\
        python rnn_generate_preprocessed.py --run 1121 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python rnn_generate_preprocessed.py --run 1122 --epoch 10 --lr 0.005 --enc-dim 32;\
        python rnn_generate_preprocessed.py --run 1123 --epoch 10 --lr 0.005 --enc-dim 64;\
        python rnn_generate_preprocessed.py --run 1124 --epoch 10 --lr 0.005 --enc-dim 128;\
        python rnn_generate_preprocessed.py --run 1125 --epoch 10 --lr 0.009 --enc-dim 256;\
        python rnn_generate_preprocessed.py --run 1126 --epoch 10 --lr 0.009 --enc-dim 512;\
        python rnn_generate_preprocessed.py --run 1127 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python rnn_generate_preprocessed.py --run 1128 --epoch 10 --lr 0.005 --dec-dim 32;\
        python rnn_generate_preprocessed.py --run 1129 --epoch 10 --lr 0.005 --dec-dim 64;\
        python rnn_generate_preprocessed.py --run 1130 --epoch 10 --lr 0.005 --dec-dim 128;\
        python rnn_generate_preprocessed.py --run 1131 --epoch 10 --lr 0.009 --dec-dim 256;\
        python rnn_generate_preprocessed.py --run 1132 --epoch 10 --lr 0.009 --dec-dim 512;\
        python rnn_generate_preprocessed.py --run 1133 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python rnn_generate_preprocessed.py --run 1134 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python rnn_generate_preprocessed.py --run 1135 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python rnn_generate_preprocessed.py --run 1136 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python rnn_generate_preprocessed.py --run 1137 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python rnn_generate_preprocessed.py --run 1138 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python rnn_generate_preprocessed.py --run 1139 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python rnn_generate_preprocessed.py --run 1140 --epoch 10 --lr 0.0005 --bs 2;\
        python rnn_generate_preprocessed.py --run 1141 --epoch 10 --lr 0.0005 --bs 8;\
        python rnn_generate_preprocessed.py --run 1142 --epoch 10 --lr 0.0005 --bs 16;\
        python rnn_generate_preprocessed.py --run 1143 --epoch 10 --lr 0.0009 --bs 32;\
        python rnn_generate_preprocessed.py --run 1144 --epoch 10 --lr 0.0009 --bs 64;\
        python rnn_generate_preprocessed.py --run 1145 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python rnn_generate_preprocessed.py --run 1146 --epoch 10 --lr 0.005 --clip 5.0;\
        python rnn_generate_preprocessed.py --run 1147 --epoch 10 --lr 0.005 --clip 4.0;\
        python rnn_generate_preprocessed.py --run 1148 --epoch 10 --lr 0.005 --clip 3.0;\
        python rnn_generate_preprocessed.py --run 1149 --epoch 10 --lr 0.009 --clip 6.0;\
        python rnn_generate_preprocessed.py --run 1150 --epoch 10 --lr 0.009 --clip 7.0;\
        python rnn_generate_preprocessed.py --run 1151 --epoch 10 --lr 0.009 --clip 10.0\
) &
