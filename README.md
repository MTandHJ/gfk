



A Simple FrameWorK of GAN.



The corresponding trained models could be downloaded [here]([A Collection of Trained DCGAN under different settings | Zenodo](https://zenodo.org/record/4759263)).



## Usage



```
┌── data # the path of data
│	├── mnist
│	├── cifar10
│	├── celeba
│	└── hdf5data # we select HDF5 file for training in default, which could be created automatically
│		├── cifar10.hdf5 # Refer to src/datasets.py for more details.
│		└── celeba.hdf5
├── gfk
│	├── freeplot # for saving image
│	├── infos # for saving trained model including generator and discriminator
│	├── logs # The curve of varying of G_loss, D_loss, FID, IS along with steps increasing
│	├── metrics # The implementation of FID and IS
│	├── models # Architectures
│	└── src
│		├── augmentation.py # some augmentations
│		├── base.py # Coach, arranging the training procdure
│		├── config.py # You can specify the ROOT and HDF5 as the path of training data.
│		├── datasets.py # how to load and save data
│		├── dict2obj.py
│		├── loadopts.py # for loading
│		├── loss_zoo.py # The implementations of bce, hinge, wgan, leastsquares loss ...
│		└── utils.py # other usful tools
└── main.py # the main file
```





## Ablation



|               Setting                | IS(⭡) | FID(⭣) | Collapse Steps |
| :----------------------------------: | :---: | :----: | :------------: |
|             DCGAN + BCE              | 2.584 | 17.95  |     48000      |
|            DCGAN + Hinge             | 2.493 | 22.78  |     52000      |
|         DCGAN + LeastSquare          | 2.57  | 20.42  |     100000     |
|            DCGAN + WLoss             | 2.672 | 17.57  |     100000     |
|           DCGAN + BCE + SN           | 2.702 | 23.16  |     74000      |
|          DCGAN + Hinge + SN          | 2.559 |  19.4  |     44000      |
|          DCGAN + BCE + EMA           | 2.613 | 15.88  |     48000      |
|         DCGAN + Hinge + EMA          | 2.578 | 17.650 |     52000      |
|           DCGAN + LS + EMA           | 2.56  | 15.92  |     100000     |
|         DCGAN + WLOSS + EMA          | 2.614 | 14.93  |     100000     |
|        DCGAN + BCE + TNormal         | 2.579 | 21.23  |     32000      |
|       DCGAN + Hinge + TNormal        | 2.579 | 18.45  |     84000      |
|         DCGAN + LS + TNormal         | 2.538 | 18.59  |     100000     |
|       DCGAN + WLOSS + TNormal        | 2.598 | 17.17  |     100000     |
|        DCGAN + BCE + Diff_Aug        | 2.601 | 15.25  |     100000     |
|       DCGAN + Hinge+ Diff_Aug        |  ???  |  ???   |      ???       |
|        DCGAN + LS + Diff_Aug         | 2.632 | 17.57  |     100000     |
|       DCGAN + WLoss + Diff_Aug       | 2.983 | 157.5  |     100000     |
|     DCGAN + Hinge + EMA + ACML=2     | 2.612 | 16.94  |     50000      |
|     DCGAN + Hinge + EMA + ACML=3     | 2.532 | 22.49  |     34000      |
|     DCGAN + Hinge + EMA + ACML=4     | 2.501 | 22.64  |     30000      |
| DCGAN + Hinge + EMA + ACML=2 + SPD=2 | 2.513 | 23.65  |     34000      |
| DCGAN + Hinge + EMA + ACML=2 + SPD=3 | 2.606 | 23.02  |     48000      |



**Note**: We choose the result before collapse if occurred.



### Steps



```
python main.py celeba --steps=100000
```



|                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210419205147561](README.assets/image-20210419205147561.png) | ![image-20210419205016148](README.assets/image-20210419205016148.png) | ![image-20210419204921420](README.assets/image-20210419204921420.png) |



![image-20210419205255525](README.assets/image-20210419205255525.png)



### Loss



|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| hinge | ![image-20210420075314509](README.assets/image-20210420075314509.png) | ![image-20210420075248362](README.assets/image-20210420075248362.png) | ![image-20210420075141332](README.assets/image-20210420075141332.png) |
|  LS   | ![image-20210420144956225](README.assets/image-20210420144956225.png) | ![image-20210420144935446](README.assets/image-20210420144935446.png) | ![image-20210420144913703](README.assets/image-20210420144913703.png) |
| wloss | ![image-20210421074309653](README.assets/image-20210421074309653.png) | ![image-20210421074252949](README.assets/image-20210421074252949.png) | ![image-20210421074235640](README.assets/image-20210421074235640.png) |



#### hinge

![image-20210420075409246](README.assets/image-20210420075409246.png)



#### LS



![image-20210420145318999](README.assets/image-20210420145318999.png)



#### WLoss

**note:** I did not impose the gradient penalty on generator or discriminator, though it seems work well.

![image-20210421074425946](README.assets/image-20210421074425946.png)



### Spectral Normalization



```
python main.py celeba -sng -snd
```



It seems that spectral normalization does little to small models, or a wrong implementation ?



|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  BCE  | ![image-20210422073126663](README.assets/image-20210422073126663.png) | ![image-20210422073110749](README.assets/image-20210422073110749.png) | ![image-20210422073055536](README.assets/image-20210422073055536.png) |
| Hinge | ![image-20210421201102175](README.assets/image-20210421201102175.png) | ![image-20210421201019447](README.assets/image-20210421201019447.png) | ![image-20210421200952582](README.assets/image-20210421200952582.png) |



#### BCE



![image-20210422073308856](README.assets/image-20210422073308856.png)



#### Hinge



![image-20210421201316021](README.assets/image-20210421201316021.png)



### EMA



EMA works so well in my sense.

```
python main.py celeba --ema
```





|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  BCE  | ![image-20210422130715257](README.assets/image-20210422130715257.png) | ![image-20210422130648382](README.assets/image-20210422130648382.png) | ![image-20210422130631905](README.assets/image-20210422130631905.png) |
| Hinge | ![image-20210422201320260](README.assets/image-20210422201320260.png) | ![image-20210422201305748](README.assets/image-20210422201305748.png) | ![image-20210422201249695](README.assets/image-20210422201249695.png) |
|  LS   | ![image-20210423084352326](README.assets/image-20210423084352326.png) | ![image-20210423084338981](README.assets/image-20210423084338981.png) | ![image-20210423084320402](README.assets/image-20210423084320402.png) |
| WLoss | ![image-20210423161016550](README.assets/image-20210423161016550.png) | ![image-20210423161001516](README.assets/image-20210423161001516.png) | ![image-20210423160948402](README.assets/image-20210423160948402.png) |



####  BCE

![image-20210422123506601](README.assets/image-20210422123506601.png)



#### Hinge



![image-20210422201157886](README.assets/image-20210422201157886.png)



#### LS



![image-20210423084433383](README.assets/image-20210423084433383.png)



#### WLoss



![image-20210423161214140](README.assets/image-20210423161214140.png)



### Truncated Normal



```
python main.py celeba --rtype=tnormal
```



|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  BCE  | ![image-20210429074950751](README.assets/image-20210429074950751.png) | ![image-20210429074928530](README.assets/image-20210429074928530.png) | ![image-20210429074912429](README.assets/image-20210429074912429.png) |
| Hinge | ![image-20210430213505472](README.assets/image-20210430213505472.png) | ![image-20210430213454208](README.assets/image-20210430213454208.png) | ![image-20210430213442150](README.assets/image-20210430213442150.png) |
|  LS   | ![image-20210501081930910](README.assets/image-20210501081930910.png) | ![image-20210501081915539](README.assets/image-20210501081915539.png) | ![image-20210501081901833](README.assets/image-20210501081901833.png) |
| WLoss | ![image-20210501214137286](README.assets/image-20210501214137286.png) | ![image-20210501214121195](README.assets/image-20210501214121195.png) | ![image-20210501214104609](README.assets/image-20210501214104609.png) |





#### BCE



![image-20210429075028547](README.assets/image-20210429075028547.png)

#### Hinge



![image-20210430213552929](README.assets/image-20210430213552929.png)



#### LS



![image-20210501081955704](README.assets/image-20210501081955704.png)



#### WLoss



![image-20210501214206303](README.assets/image-20210501214206303.png)



### Diff_Aug



```
python main.py celeba --aug_polic=diff_aug
```



|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  BCE  | ![image-20210502083342466](README.assets/image-20210502083342466.png) | ![image-20210502083327692](README.assets/image-20210502083327692.png) | ![image-20210502083316058](README.assets/image-20210502083316058.png) |
| Hinge | ![image-20210502205525963](README.assets/image-20210502205525963.png) | ![image-20210502205513645](README.assets/image-20210502205513645.png) | ![image-20210502205500940](README.assets/image-20210502205500940.png) |
|  LS   | ![image-20210503073829554](README.assets/image-20210503073829554.png) | ![image-20210503073805305](README.assets/image-20210503073805305.png) | ![image-20210503073740785](README.assets/image-20210503073740785.png) |
| WLOSS | ![image-20210506141517974](README.assets/image-20210506141517974.png) | ![image-20210506141458007](README.assets/image-20210506141458007.png) | ![image-20210506141443045](README.assets/image-20210506141443045.png) |



#### BCE



![image-20210502083421929](README.assets/image-20210502083421929.png)



#### Hinge



![image-20210502205703051](README.assets/image-20210502205703051.png)



#### LS



![image-20210503073907020](README.assets/image-20210503073907020.png)

#### WLoss



![image-20210506141417491](README.assets/image-20210506141417491.png)



### acml



```
python main.py celeba -cg=hinge -cd=hinge --ema -acml=2
```





|        |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| acml=2 | ![image-20210510212920238](README.assets/image-20210510212920238.png) | ![image-20210510212935268](README.assets/image-20210510212935268.png) | ![image-20210510212948980](README.assets/image-20210510212948980.png) |
| acml=3 | ![image-20210511083810373](README.assets/image-20210511083810373.png) | ![image-20210511083756248](README.assets/image-20210511083756248.png) | ![image-20210511083739968](README.assets/image-20210511083739968.png) |
| acml=4 | ![image-20210512104858329](README.assets/image-20210512104858329.png) | ![image-20210512104843396](README.assets/image-20210512104843396.png) | ![image-20210512104803697](README.assets/image-20210512104803697.png) |



#### acml=2



![image-20210510213039982](README.assets/image-20210510213039982.png)

#### acml=3



![image-20210511083842488](README.assets/image-20210511083842488.png)



#### acml=4



![image-20210512104931245](README.assets/image-20210512104931245.png)



### steps_per_D



```
python main.py celeba -cg=hinge -cd=hinge --ema -acml=2 -spd=2
```





|       |                             Loss                             |                            IS(⭡)                             |                            FID(⭣)                            |
| :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| spd=2 | ![image-20210513082354592](README.assets/image-20210513082354592.png) | ![image-20210513082342241](README.assets/image-20210513082342241.png) | ![image-20210513082328623](README.assets/image-20210513082328623.png) |
| spd=3 | ![image-20210514084139130](README.assets/image-20210514084139130.png) | ![image-20210514084127282](README.assets/image-20210514084127282.png) | ![image-20210514084115967](README.assets/image-20210514084115967.png) |





#### spd=2



![image-20210513082456239](README.assets/image-20210513082456239.png)





#### spd=3



![image-20210514084236909](README.assets/image-20210514084236909.png)

