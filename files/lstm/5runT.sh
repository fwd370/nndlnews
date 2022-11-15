#!/bin/sh

nohup $(/
	python lstm_generate.py --run 510 --epoch 10 --lr 0.005 --h-dim 32;\
	python lstm_generate.py --run 511 --epoch 10 --lr 0.005 --h-dim 64;\
	python lstm_generate.py --run 512 --epoch 10 --lr 0.005 --h-dim 128;\
	python lstm_generate.py --run 513 --epoch 10 --lr 0.009 --h-dim 256;\
        python lstm_generate.py --run 514 --epoch 10 --lr 0.009 --h-dim 512;\
        python lstm_generate.py --run 515 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python lstm_generate.py --run 516 --epoch 10 --lr 0.0005 --h-dim 32;\
        python lstm_generate.py --run 517 --epoch 10 --lr 0.0005 --h-dim 64;\
        python lstm_generate.py --run 518 --epoch 10 --lr 0.0005 --h-dim 128;\
        python lstm_generate.py --run 519 --epoch 10 --lr 0.0009 --h-dim 256;\
        python lstm_generate.py --run 520 --epoch 10 --lr 0.0009 --h-dim 512;\
        python lstm_generate.py --run 521 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python lstm_generate.py --run 522 --epoch 10 --lr 0.005 --enc-dim 32;\
        python lstm_generate.py --run 523 --epoch 10 --lr 0.005 --enc-dim 64;\
        python lstm_generate.py --run 524 --epoch 10 --lr 0.005 --enc-dim 128;\
        python lstm_generate.py --run 525 --epoch 10 --lr 0.009 --enc-dim 256;\
        python lstm_generate.py --run 526 --epoch 10 --lr 0.009 --enc-dim 512;\
        python lstm_generate.py --run 527 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python lstm_generate.py --run 528 --epoch 10 --lr 0.005 --dec-dim 32;\
        python lstm_generate.py --run 529 --epoch 10 --lr 0.005 --dec-dim 64;\
        python lstm_generate.py --run 530 --epoch 10 --lr 0.005 --dec-dim 128;\
        python lstm_generate.py --run 531 --epoch 10 --lr 0.009 --dec-dim 256;\
        python lstm_generate.py --run 532 --epoch 10 --lr 0.009 --dec-dim 512;\
        python lstm_generate.py --run 533 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python lstm_generate.py --run 534 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python lstm_generate.py --run 535 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python lstm_generate.py --run 536 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python lstm_generate.py --run 537 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python lstm_generate.py --run 538 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python lstm_generate.py --run 539 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python lstm_generate.py --run 540 --epoch 10 --lr 0.0005 --bs 2;\
        python lstm_generate.py --run 541 --epoch 10 --lr 0.0005 --bs 8;\
        python lstm_generate.py --run 542 --epoch 10 --lr 0.0005 --bs 16;\
        python lstm_generate.py --run 543 --epoch 10 --lr 0.0009 --bs 32;\
        python lstm_generate.py --run 544 --epoch 10 --lr 0.0009 --bs 64;\
        python lstm_generate.py --run 545 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python lstm_generate.py --run 546 --epoch 10 --lr 0.005 --clip 5.0;\
        python lstm_generate.py --run 547 --epoch 10 --lr 0.005 --clip 4.0;\
        python lstm_generate.py --run 548 --epoch 10 --lr 0.005 --clip 3.0;\
        python lstm_generate.py --run 549 --epoch 10 --lr 0.009 --clip 6.0;\
        python lstm_generate.py --run 550 --epoch 10 --lr 0.009 --clip 7.0;\
        python lstm_generate.py --run 551 --epoch 10 --lr 0.009 --clip 10.0\
) &
