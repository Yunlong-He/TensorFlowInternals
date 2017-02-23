
**介绍**

这里使用<a href="https://github.com/tensorflow/models">model zoo</a>的代码在tensorflow上进行训练

**数据下载**

虽然model本身提供了下载imagenet数据的脚本，但是我之前已经下载过，所以直接开始预处理就可以了。

预处理，生成bounding_box文件
```sh
$ ls /data0/data/image-net/ILSVRC/
Annotations  Data  ImageSets
$ python process_bounding_boxes.py /data0/data/image-net/ILSVRC/Annotations/CLS-LOC/train ./imagenet_lsvrc_2015_synsets.txt | sort > /data0/data/image-net/processed/imagenet_2015_bounding_boxes.csv
```
生成TF-Records
```sh
cd inception/inception/data/
python build_imagenet_data.py --train_directory=/data0/data/image-net/ILSVRC/Data/CLS-LOC/train --validation_directory=/data0/data/image-net/ILSVRC/Data/CLS-LOC/val --output_directory=/data0/data/image-net/processed --imagenet_metadata_file=./imagenet_metadata.txt --labels_file=./imagenet_lsvrc_2015_synsets.txt --bounding_box_file=./imagenet_2015_bounding_boxes.csv
```

开始训练
```sh
PYTHONPATH=. python inception/imagenet_train.py --num_gpus=1 --batch_size=32 --train_dir=tt --data_dir=/data0/data/image-net/processed
````

我们的任务是提交到yarn集群上进行的，训练任务最终是跑在docker里，TF-Records文件是通过NFS来访问的。简单测试了一下inception的训练过程，性能略微有些损失，但非常小，大概在1%左右