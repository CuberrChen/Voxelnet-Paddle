# **VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection**
## **VoxelNet：基于点云的三维空间信息逐层次学习网络**

# **1、项目总览**
## **①、简介**
本项目主要是对来自2017年苹果公司基于点云的3D物体检测论文"VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"进行复现。
VoxelNet只利用点云数据，在无人驾驶环境下实现了高精度的三维物体检测。

Note：**项目目前可以训练、可以评估预测**。但是（单卡训练，bs=2,epoch到100）未达到论文精度（论文bs=16,epoch到160）。有兴趣的同学可以自己尝试再进行复现。 

另外**重要的是**：训练存在内存泄漏问题，**之前定位到dataloader泄漏**，不知道是数据预处理部分有问题还是paddle的问题，但是自己没功夫检查和修改了，具体位置有兴趣的自己排查吧。
不然就得训练一段时间断掉再resume，其中train_fix_oom.py就是采用了读取内存，超过阈值resume的方案～

想要复现指标，一个是要调整参数（主要是yaml文件里的参数，但是参数改动可能造成不稳定），一个是要把项目里的bug fix掉噢！我认为**调大batchsize应该会有大的提升**！但是我并未尝试过多卡训练，所以可能会出问题。

**ps**：本项目主要是从second修改而来，所以稍加修改即可复现second和pointpillar。这是目前唯一能找到的可以训练和评估voxelnet paddle版本噢～

之前训练过程中(未训练完)的某次结果指标：


```
Car AP@0.70, 0.70, 0.70:
bbox AP:53.40, 41.52, 35.37
bev  AP:52.85, 40.87, 34.85
3d   AP:50.86, 36.90, 30.71
aos  AP:28.13, 23.91, 21.17
Car AP@0.70, 0.50, 0.50:
bbox AP:53.40, 41.52, 35.37
bev  AP:53.44, 42.09, 35.77
3d   AP:53.42, 41.90, 35.65
aos  AP:28.13, 23.91, 21.17
```



# **2、网络简介**

将三维点云划分为一定数量的Voxel，经过点的随机采样以及归一化后，对每一个非空Voxel使用若干个VFE(Voxel Feature Encoding)层进行局部特征提取，得到Voxel-wise Feature，然后经过3D Convolutional Middle Layers进一步抽象特征（增大感受野并学习几何空间表示），最后使用RPN(Region Proposal Network)对物体进行分类检测与位置回归。VoxelNet整个pipeline如下图所示。

## **①、网络结构**
![](https://pic2.zhimg.com/80/v2-557dad554081f49cc582d51b94a88b99_720w.jpg)

通过层叠的VFE层将体素编码，然后3D卷积进一步放大局部voxel特征，将点云转化成高维的体积的表达。最后通过RPN产生检测结果。

## **②、特征提取模块：VFE**

![](https://pic4.zhimg.com/80/v2-ebc7bab54be02a54bb8ca007e1053e8f_720w.jpg)
## **3、RPN模块**

![](https://img-blog.csdnimg.cn/20210329193322806.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JyaWJsdWU=,size_16,color_FFFFFF,t_70#pic_center)

论文中提到，RPN 中的 FCN网络分为 3 块，每一块都会实现 2x 效果的下采样率。然后，又实现了向上采样，将倒数 3 块上采样到固定的尺寸，然后拼接起来。最终，由上采样拼接后的卷积引出 2 个目标分支：概率图和回归图
注意它们的尺寸，概率图通道数是 2，代表正负 anchor 的概率，这个概率应该通过 softmax 处理过。
回归图的通道数是 7，代表的就是一个 anchor 的 3D 信息(x,y,z,l,w,h,theta)

## **②、损失函数**

![](https://img-blog.csdnimg.cn/20210329200710185.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JyaWJsdWU=,size_16,color_FFFFFF,t_70#pic_center)

总体 Loss 由 2 部分组成：
- 分类 Loss
- 回归 Loss

# **3、网络训练**

## **1、数据集的准备(十几分钟解压)**
![](https://ai-studio-static-online.cdn.bcebos.com/446b20a01f7b4c9cb168c329a54f22d9e9d5d44acddf40de80e74d73ba7e812f)<br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>VoxelNet网络在KITTI数据集中的3D Detection数据集上面进行训练，数据集中包含7481张训练图片以及7518张测试图片，一共有80256个标记物体，并且测试模式包含普通的视角以及鸟瞰视角。</font>


```python
!rm -rf kitti/
!mkdir -p kitti/training/velodyne_reduced
!mkdir -p kitti/testing/velodyne_reduced
```


```python
!unzip data/data50186/data_object_calib.zip -d kitti/
```

     extracting: kitti/testing/calib/004455.txt  


```python
!unzip data/data50186/image_training.zip -d kitti/training/
!unzip data/data50186/data_object_label_2.zip -d kitti/training/
!unzip data/data50186/velodyne_training_1.zip -d kitti/training/
!unzip data/data50186/velodyne_training_2.zip -d kitti//training/
!unzip data/data50186/velodyne_training_3.zip -d kitti/training/
```

      inflating: kitti/training/velodyne_training_3/007480.bin  


```python
!unzip data/data50186/image_testing.zip -d kitti/testing/
!unzip data/data50186/velodyne_testing_1.zip -d kitti/testing/
!unzip data/data50186/velodyne_testing_2.zip -d kitti/testing/
!unzip data/data50186/velodyne_testing_3.zip -d kitti/testing/
```

      inflating: kitti/testing/velodyne_testing_3/007517.bin  


```python
!mv kitti/training/training/* kitti/training/
!rm -rf kitti/training/training/
!mv kitti/testing/testing/* kitti/testing/
!rm -rf kitti/testing/testing/
```


```python
!mkdir kitti/training/velodyne
!mv kitti/training/velodyne_training_1/* kitti/training/velodyne/
!mv kitti/training/velodyne_training_2/* kitti/training/velodyne/
!mv kitti/training/velodyne_training_3/* kitti/training/velodyne/
!rm -rf kitti/training/velodyne_training_1
!rm -rf kitti/training/velodyne_training_2
!rm -rf kitti/training/velodyne_training_3
!mkdir kitti/testing/velodyne
!mv kitti/testing/velodyne_testing_1/* kitti/testing/velodyne
!mv kitti/testing/velodyne_testing_2/* kitti/testing/velodyne
!mv kitti/testing/velodyne_testing_3/* kitti/testing/velodyne
!rm -rf kitti/testing/velodyne_testing_1
!rm -rf kitti/testing/velodyne_testing_2
!rm -rf kitti/testing/velodyne_testing_3
```

## **2、安装必要的库**


```python
!pip install shapely pybind11 protobuf scikit-image pillow fire scikit-image memory_profiler psutil
!pip install numpy==1.17
!pip install numba==0.48.0
```


## **3、数据集处理与准备**
&emsp;&emsp;&emsp;&emsp;<font size=4>对KITTI数据集进行处理。</font><br><br>

#### 数据集应有结构

```
kitti/
├── training/
├──    	├── calib
├──    	├── image_2
├──    	├── label_2
├──    	├── velodyne
├──    	└── velodyne_reduced
├── testing/
├──    	├── calib
├──    	├── image_2
├──    	├── velodyne
├──    	└── velodyne_reduced
├── gt_database/
		├── 4264_Car_1.bin
    	...
├── kitti_dbinfos_train.pkl
├── kitti_infos_test.pkl
├── kitti_infos_train.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_val.pkl

├── test.txt
├── train.txt
├── trainval.txt
└── val.txt
```


```python
%cd VoxelNet-Paddle/
```

创建存放数据信息的文件：

这之前可能需要解压VoxelNet-Paddle/ImageSets.zip获取trainval.txt等几个split文件，放在kitti文件夹下面。

```python
!python create_data.py create_kitti_info_file --data_path=/home/aistudio/kitti ## 报错的话，可能需要修改create_data.py内的路径
```
    Kitti info train file is saved to /home/aistudio/kitti/kitti_infos_train.pkl
    Kitti info val file is saved to /home/aistudio/kitti/kitti_infos_val.pkl
    Kitti info trainval file is saved to /home/aistudio/kitti/kitti_infos_trainval.pkl
    Kitti info test file is saved to /home/aistudio/kitti/kitti_infos_test.pkl


```python
!python create_data.py create_reduced_point_cloud --data_path=/home/aistudio/kitti
```

    [100.0%][===================>][31.37it/s][01:54>00:00]     
    [100.0%][===================>][30.13it/s][01:48>00:00]   
    [100.0%][===================>][32.45it/s][03:35>00:00]   


```python
!python create_data.py create_groundtruth_database --data_path=/home/aistudio/kitti
```

    [100.0%][===================>][142.03it/s][00:37>00:00]    
    load 14357 Car database infos
    load 2207 Pedestrian database infos
    load 734 Cyclist database infos
    load 1297 Van database infos
    load 56 Person_sitting database infos
    load 488 Truck database infos
    load 224 Tram database infos
    load 337 Misc database infos

## **4、训练**
在数据集准备好以后，项目结构应该是：
```
├── kitti/
├── VoxelNet-Paddle/
```

然后就可以进入到VoxelNet-Paddle/进行训练/评估。

第一个训练指令没有vdl可视化噢～

第二个训练指令加入训练可视化以及多GPU支持～（因为自己没有多卡，所有没测试过呢，或许多卡训练能够还原指标噢！！！）！

```python
# !python train.py train --cfg_file=configs/voxelnet_kitti_car.yaml --model_dir=./output
#或者
!python train_mgpu.py --config=configs/voxelnet_kitti_car.yaml --model_dir=./output --use_vdl=True
```

    ^C

## **5、评估**

可下载之前训练的模型进行评估测试，注意修改model_dir路径噢～
- [网盘链接](https://pan.baidu.com/s/1mV9GFoFlk5upgxW517ACmA): 提取码: 75pe

```python
!python eval.py eval --cfg_file=configs/voxelnet_kitti_car.yaml --model_dir=./output
```

    ^C

# **6、Reference**
&emsp;&emsp;&emsp;&emsp;<font size=4>[论文](https://arxiv.org/pdf/1711.06396.pdf)</font><br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>[traveller59/second.pytorch](https://github.com/traveller59/second.pytorch)</font><br><br>
&emsp;&emsp;&emsp;&emsp;<font size=4>[叶月火狐/PointPillars](https://aistudio.baidu.com/aistudio/projectdetail/2250701?channelType=0&channel=0)</font><br><br>

