
## 编译安装TensorFlow

### 准备编译环境
编译过程参考
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md

首先用git下载源代码
```sh
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
```

刚下载下来的时候，代码处于master的最新版本上，master版本不一定是稳定版本，所以建议选择合适的branch或者tag来做测试。早期开始测试的时候0.8还没有release，所以用了0.7.1的版本，但是现在1.0已经release了，可以用1.0的版本来编译。
```sh
$ git checkout -b branch-v0.7.1 v0.7.1
```

编译Tensorflow需要的开发环境包括python，pip, numpy，比较特殊的是Tensorflow还依赖bazel和swig，bazel是google开源的编译框架，据说google内部很多项目都用bazel编译。建议大家除了安装bazel，最好也学习一下bazel的工作原理，这样才能比较容易的看懂tensorflow的源代码结构。至于swig，这是一个工具，可以把c++的库封装成其他语言的接口，比如python, java等。当然tensorflow里只提供了python的接口，如果我们需要在spark里使用tensorflow，或许可以通过swig来生成java接口。
```sh
$ apt-get install python-numpy swig python-dev

for centos, you can use yum install
$ yum install python-numpy swig python-dev
```

bazel可以通过访问http://bazel.io/来了解怎么安装，上面还有比较详细的教程。 一个比较简便的方法是下载self-contained的源代码进行编译

如果你的机器上有nvidia的GPU，那么建议你安装cuda的toolkit和cudnn，网上这方面的教程比较多，这里暂时不做介绍。

###开始编译TensorFlow
```sh
$ ./configure  
```

在这里需要选择编译的各种选项，由于python2.7和python3.4可以共存，如果是需要用Python3来运行tensorflow，注意选择合适的python路径。如果前面安装了cuda和cudnn，在这里也可以选择支持cuda，一般使用缺省的路径即可。

开始编译tensorflow，生成pip package
```sh
$ bazel build -c dbg --config=cuda //tensorflow/tools/pip_package:build_pip_package
```
需要注意的是：
	1. 需要编译debug版本的话，加上参数 -c dbg即可
	2. 需要支持cuda，加上参数--config=cuda
	3. 编译过程中需要访问google自己host的一些项目（boringssl），如果在墙内没办法编译，还是建议选择google编译好的版本。如果实在需要编译源代码，可以尝试修改bazel文件,避免依赖boringssl，应该也可以继续编译，不过我没有尝试。

生成whale安装包，注意最后生成的安装包的名字是和你的编译选项有关系的
tensorflow-0.7.1-cp27-none-linux_x86_64.whl    -  python2.7  64位
tensorflow-0.7.1-cp34-cp34m-linux_x86_64.whl   -  python3.4  64位
tensorflow-0.7.1-py3-none-any.whl							 -  python3.4  64位 带debug信息
```sh
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo -E pip3 install /tmp/tensorflow_pkg/tensorflow-0.7.1-py3-none-any.whl
```

###测试TensorFlow

接下来可以运行mnist的例子来测试TensorFlow
```sh
$ cd tensorflow/models/image/mnist
$ python convolutional.py

....

Step 8400 (epoch 9.77), 23.7 ms
Minibatch loss: 1.598, learning rate: 0.006302
Minibatch error: 0.0%
Validation error: 0.8%
Step 8500 (epoch 9.89), 23.5 ms
Minibatch loss: 1.605, learning rate: 0.006302
Minibatch error: 1.6%
Validation error: 0.8%
Test error: 0.9%
```

这样就表示TensorFlow可以正常使用了。


**FAQ**

1. error: invalid command 'bdist_wheel'
   
   可以通过升级setuptools来解决。参考http://www.cnblogs.com/BugQiang/archive/2015/08/22/4732991.html
   
```sh
   pip install wheel
   pip install setuptools --upgrade
```

2. Python.h not found

    这个比较特殊，我的机器上单独安装了一个python-devel的库，但是不是使用系统命令安装的，编译时无法找到。这个时候单独安装Python-devel就好了。
