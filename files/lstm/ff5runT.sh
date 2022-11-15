#!/bin/sh

nohup $(/
	python lstm_generate_full.py --run 710 --epoch 10 --lr 0.005 --h-dim 32;\
	python lstm_generate_full.py --run 711 --epoch 10 --lr 0.005 --h-dim 64;\
	python lstm_generate_full.py --run 712 --epoch 10 --lr 0.005 --h-dim 128;\
	python lstm_generate_full.py --run 713 --epoch 10 --lr 0.009 --h-dim 256;\
        python lstm_generate_full.py --run 714 --epoch 10 --lr 0.009 --h-dim 512;\
        python lstm_generate_full.py --run 715 --epoch 10 --lr 0.009 --h-dim 1024\
) &

nohup $(
	python lstm_generate_full.py --run 716 --epoch 10 --lr 0.0005 --h-dim 32;\
        python lstm_generate_full.py --run 717 --epoch 10 --lr 0.0005 --h-dim 64;\
        python lstm_generate_full.py --run 718 --epoch 10 --lr 0.0005 --h-dim 128;\
        python lstm_generate_full.py --run 719 --epoch 10 --lr 0.0009 --h-dim 256;\
        python lstm_generate_full.py --run 720 --epoch 10 --lr 0.0009 --h-dim 512;\
        python lstm_generate_full.py --run 721 --epoch 10 --lr 0.0009 --h-dim 1024\
) &

nohup $(
	python lstm_generate_full.py --run 722 --epoch 10 --lr 0.005 --enc-dim 32;\
        python lstm_generate_full.py --run 723 --epoch 10 --lr 0.005 --enc-dim 64;\
        python lstm_generate_full.py --run 724 --epoch 10 --lr 0.005 --enc-dim 128;\
        python lstm_generate_full.py --run 725 --epoch 10 --lr 0.009 --enc-dim 256;\
        python lstm_generate_full.py --run 726 --epoch 10 --lr 0.009 --enc-dim 512;\
        python lstm_generate_full.py --run 727 --epoch 10 --lr 0.009 --enc-dim 1024\
) &

nohup $(
	python lstm_generate_full.py --run 728 --epoch 10 --lr 0.005 --dec-dim 32;\
        python lstm_generate_full.py --run 729 --epoch 10 --lr 0.005 --dec-dim 64;\
        python lstm_generate_full.py --run 730 --epoch 10 --lr 0.005 --dec-dim 128;\
        python lstm_generate_full.py --run 731 --epoch 10 --lr 0.009 --dec-dim 256;\
        python lstm_generate_full.py --run 732 --epoch 10 --lr 0.009 --dec-dim 512;\
        python lstm_generate_full.py --run 733 --epoch 10 --lr 0.009 --dec-dim 1024\
) &

nohup $(
	python lstm_generate_full.py --run 734 --epoch 10 --lr 0.005 --dec-dim 32 --enc-dim 32;\
        python lstm_generate_full.py --run 735 --epoch 10 --lr 0.005 --dec-dim 64 --enc-dim 64;\
        python lstm_generate_full.py --run 736 --epoch 10 --lr 0.005 --dec-dim 128 --enc-dim 128;\
        python lstm_generate_full.py --run 737 --epoch 10 --lr 0.009 --dec-dim 256 --enc-dim 256;\
        python lstm_generate_full.py --run 738 --epoch 10 --lr 0.009 --dec-dim 512 --enc-dim 512;\
        python lstm_generate_full.py --run 739 --epoch 10 --lr 0.009 --dec-dim 1024 --enc-dim 1024\
) &

nohup $(
        python lstm_generate_full.py --run 740 --epoch 10 --lr 0.0005 --bs 2;\
        python lstm_generate_full.py --run 741 --epoch 10 --lr 0.0005 --bs 8;\
        python lstm_generate_full.py --run 742 --epoch 10 --lr 0.0005 --bs 16;\
        python lstm_generate_full.py --run 743 --epoch 10 --lr 0.0009 --bs 32;\
        python lstm_generate_full.py --run 744 --epoch 10 --lr 0.0009 --bs 64;\
        python lstm_generate_full.py --run 745 --epoch 10 --lr 0.0009 --bs 128\
) &

nohup $(
        python lstm_generate_full.py --run 746 --epoch 10 --lr 0.005 --clip 5.0;\
        python lstm_generate_full.py --run 747 --epoch 10 --lr 0.005 --clip 4.0;\
        python lstm_generate_full.py --run 748 --epoch 10 --lr 0.005 --clip 3.0;\
        python lstm_generate_full.py --run 749 --epoch 10 --lr 0.009 --clip 6.0;\
        python lstm_generate_full.py --run 750 --epoch 10 --lr 0.009 --clip 7.0;\
        python lstm_generate_full.py --run 751 --epoch 10 --lr 0.009 --clip 10.0\
) &
