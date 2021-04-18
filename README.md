

A Simple FrameWorK of GAN.


## Usage



``` python
python gan.py celeba -h5 --dim_latent=100 \
	-cg=bce -lrg=0.0002 -sng --ema -spg=1 --rtype=normal \
    -cd=bce -lrd=0.0002 -snd --aug_policy=null -spd=1 \
    --steps=8000 -b=128
```