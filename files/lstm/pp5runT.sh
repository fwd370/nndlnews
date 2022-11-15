#!/bin/sh

nohup $(/
	python lstm_generate_preprocessed.py --run 610 --epoch 10 --lr 0.005 --h-dim 32;\
	python lstm_generate_preprocessed.py --run 611 --epoch 10 --lr 0.005 --h-dim 64;\
	python lstm_generate_preprocessed.py --run 612 --epoch 10 --lr 0.005 --h-dim 128;\
	python lstm_generate_preprocessed.py --run 613 --epoch 10 --lr 0.009 --h-dim 256;\
        python lstm_generate_preprocessed.py --run 614 --epoch 10 --lr 0.009 --h-dim 512;\
        python lstm_generate_preprocessed.py --run 615 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python lstm_generate_preprocessed.py --run 616 --epoch 10 --lr 0.0005 --h-dim 32;\
        python lstm_generate_preprocessed.py --run 617 --epoch 10 --lr 0.0005 --h-dim 64;\
        python lstm_generate_preprocessed.py --run 618 --epoch 10 --lr 0.0005 --h-dim 128;\
        python lstm_generate_preprocessed.py --run 619 --epoch 10 --lr 0.0009 --h-dim 256;\
        python lstm_generate_preprocessed.py --run 620 --epoch 10 --lr 0.0009 --h-dim 512;\
        python lstm_generate_preprocessed.py --run 621 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python lstm_generate_preprocessed.py --run 622 --epoch 10 --lr 0.005 --enc-dim 32;\
        python lstm_generate_preprocessed.py --run 623 --epoch 10 --lr 0.005 --enc-dim 64;\
        python lstm_generate_preprocessed.py --run 624 --epoch 10 --lr 0.005 --enc-dim 128;\
        python lstm_generate_preprocessed.py --run 625 --epoch 10 --lr 0.009 --enc-dim 256;\
        python lstm_generate_preprocessed.py --run 626 --epoch 10 --lr 0.009 --enc-dim 512;\
        python lstm_generate_preprocessed.py --run 627 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python lstm_generate_preprocessed.py --run 628 --epoch 10 --lr 0.005 --dec-dim 32;\
        python lstm_generate_preprocessed.py --run 629 --epoch 10 --lr 0.005 --dec-dim 64;\
        python lstm_generate_preprocessed.py --run 630 --epoch 10 --lr 0.005 --dec-dim 128;\
        python lstm_generate_preprocessed.py --run 631 --epoch 10 --lr 0.009 --dec-dim 256;\
        python lstm_generate_preprocessed.py --run 632 --epoch 10 --lr 0.009 --dec-dim 512;\
        python lstm_generate_preprocessed.py --run 633 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python lstm_generate_preprocessed.py --run 634 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python lstm_generate_preprocessed.py --run 635 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python lstm_generate_preprocessed.py --run 636 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python lstm_generate_preprocessed.py --run 637 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python lstm_generate_preprocessed.py --run 638 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python lstm_generate_preprocessed.py --run 639 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python lstm_generate_preprocessed.py --run 640 --epoch 10 --lr 0.0005 --bs 2;\
        python lstm_generate_preprocessed.py --run 641 --epoch 10 --lr 0.0005 --bs 8;\
        python lstm_generate_preprocessed.py --run 642 --epoch 10 --lr 0.0005 --bs 16;\
        python lstm_generate_preprocessed.py --run 643 --epoch 10 --lr 0.0009 --bs 32;\
        python lstm_generate_preprocessed.py --run 644 --epoch 10 --lr 0.0009 --bs 64;\
        python lstm_generate_preprocessed.py --run 645 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python lstm_generate_preprocessed.py --run 646 --epoch 10 --lr 0.005 --clip 5.0;\
        python lstm_generate_preprocessed.py --run 647 --epoch 10 --lr 0.005 --clip 4.0;\
        python lstm_generate_preprocessed.py --run 648 --epoch 10 --lr 0.005 --clip 3.0;\
        python lstm_generate_preprocessed.py --run 649 --epoch 10 --lr 0.009 --clip 6.0;\
        python lstm_generate_preprocessed.py --run 650 --epoch 10 --lr 0.009 --clip 7.0;\
        python lstm_generate_preprocessed.py --run 651 --epoch 10 --lr 0.009 --clip 10.0\
) &
