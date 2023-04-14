# MFRAN-PyTorch

[Image super-resolution with multi-scale fractal residual attention network]([[vanbou/MFRAN (github.com)](https://github.com/vanbou/MFRAN))), Xiaogang Song, Wanbo Liu, Li Liang, Weiwei Shi, Guo Xie, Xiaofeng Lu, Xinhong Hei

## Introduction

- **src/data** are used to process the dataset

- **src/loss** stores the loss function
- **src/model** sotres the proposed model and the tool classes for calculating the number of parameters
- **src/main.py** is the main function
- **src/option.py** is set for the training/test parameter
- **src/template.py** provides training/testing templates
- **src/trainer.py** is the training code
- **src/utility.py** is a utility class
- **src/videotester.py** is used to process video

In **test/**, PSNR and SSIM are used to test SR image, and can also be used for data set generation. The running environment is matlab

## Dependencies

* Python 3.6

* PyTorch >= 1.7

* numpy

* skimage

* **imageio**

* matplotlib

* tqdm

* cv2 

* torchstat (model params statistics)

  

## Prepare work
Clone this repository into any place you want.
```bash
git clone https://github.com/vanbou/MFRAN
```

You can evaluate your models with widely-used benchmark datasets:

For these datasets, we first convert the result images to YCbCr color space and evaluate PSNR on the Y channel only. You can download [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB), you can also download from https://pan.baidu.com/s/1iX46n5fdNix3J0ANN0FItg Extract code：49mx. 

Set ``--dir_data <where_benchmark_folder_located>`` to evaluate the MFRAN with the benchmarks.  

We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```src/option.py``` to the place where DIV2K images are located.

We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

If you have enough RAM (>= 32GB), you can use ``--ext bin`` argument to pack all DIV2K images in one binary file.



## How  To Train

```python
cd src   
```

For ×2

```python
python main.py --scale 2 --save MFRAN_x2 --model MFRAN --epoch 1000 --batch_size 16 --patch_size 96
```

For ×3

```python
python main.py --scale 3 --save MFRAN_x3 --model MFRAN --epoch 1000 --batch_size 16 --patch_size 144
```

For ×4

```python
python main.py --scale 4 --save MFRAN_x4 --model MFRAN --epoch 1000 --batch_size 16 --patch_size 192
```



## How  To Test

You can download pretrain model from 

https://pan.baidu.com/s/1AdqwG6y1CCi0rJk8_zjrTQ 
Extract code：mjnp

You can test our super-resolution algorithm with your images. Place your images in ``test`` folder. (like ``test/<your_image>``) We support **png** and **jpeg** files.

```bash
cd src    
python main.py --template MFRAN_test --data_test Set5+Set14+B100+Urban100+Manga109 --save MFRAN_x2_result --pre_train weight/MFRAN-2x.pt
```

or

```python
python main.py --data_test Set5+Set14+B100+Urban100+Manga109  --scale 4 --pre_train 'pretrain model path' --test_only  --chop
```

if you want to test DIV2K, add --data_range 801-900, if you need test self-ensemble method, add --self_ensemble

You can find the result images from ```experiment/``` folder.

Meanwhile，you can test your results with another method，which is based RCAN：

> cd test
>
> run Evaluate_PSNR_SSIM.m

If you want to calculate the number of model parameters, execute the following command:

```python
cd src/model
python cal_params.py
```

To test the number of MFRAN parameters for different scaling factors, change  **--scale** in **option.py**

## Results

We have published test records, detailed in **results/**

you can download our results form here:

https://pan.baidu.com/s/1sjf0TnQh-IwY33L5tSijxw 
Extract code：40y9



## Appreciate

Our work is based on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) and [RCAN]](https://github.com/yulunzhang/RCAN) . Thank you for your contributions
