



A Simple FrameWorK of GAN.



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



|       Setting       | IS(⭡) | FID(⭣) | Collapse Steps |
| :-----------------: | :---: | :----: | :------------: |
|     DCGAN + BCE     | 2.584 | 17.95  |     48000      |
|    DCGAN + Hinge    | 2.493 | 22.78  |     52000      |
| DCGAN + LeastSquare | 2.57  | 20.42  |     100000     |
|    DCGAN + WLoss    | 2.672 | 17.57  |     100000     |
|  DCGAN + BCE + SN   | 2.702 | 23.16  |     74000      |
| DCGAN + Hinge + SN  | 2.559 |  19.4  |     44000      |
|  DCGAN + BCE + EMA  | 2.613 | 15.88  |     48000      |
| DCGAN + Hinge + EMA | 2.578 | 17.650 |     52000      |
|  DCGAN + LS + EMA   | 2.56  | 15.92  |     100000     |
| DCGAN + WLOSS + EMA | 2.614 | 14.93  |     100000     |
|                     |       |        |                |



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



### Hinge



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



|      |      |      |      |
| :--: | :--: | :--: | :--: |
|      |      |      |      |

