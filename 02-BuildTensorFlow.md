
## 在CentOS 7 下编译安装TensorFlow

编译过程参考
https://www.tensorflow.org/install/install_sources

### 准备编译环境

编译Tensorflow需要的开发环境包括jdk-1.8, python, pip, numpy等
```sh
$ yum install java-1.8.0-openjdk-devel python-devel python-numpy swig python-dev python-pip
```

比较特殊的是Tensorflow还依赖bazel和swig，bazel是google开源的编译框架，据说google内部很多项目都用bazel编译。建议大家除了安装bazel，最好也学习一下bazel的工作原理，这样才能比较容易的看懂tensorflow的源代码结构。至于swig，这是一个工具，可以把c++的库封装成其他语言的接口，比如python, java等。当然tensorflow里只提供了python的接口，如果我们需要在spark里使用tensorflow，或许可以通过swig来生成java接口。
参考<a href="https://bazel.build/versions/master/docs/install.html">这里</a>安装bazel

下载最新的bazel https://github.com/bazelbuild/bazel/releases。我这里使用--user选项把bazel安装在$HOME/bin下，之前也把$HOME/bin
加到我的PATH里了。

```sh
$ chmod +x bazel-version-installer-os.sh
$ ./bazel-version-installer-os.sh --user
```

首先用git下载源代码，刚下载下来的时候，代码处于master的最新版本上，master版本不一定是稳定版本，所以建议选择合适的branch或者tag来做测试。早期开始测试的时候0.8还没有release，所以用了0.7.1的版本，但是现在1.0已经release了，可以用1.0的版本来编译。

```sh
[hyl@localhost lab]$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
[hyl@localhost lab]$ cd tensorflow
[hyl@localhost tensorflow]$ git checkout -b r1.0 remotes/origin/r1.0

```

如果你的机器上有nvidia的GPU，那么建议你安装cuda的toolkit和cudnn，网上这方面的教程比较多，这里暂时不做介绍。
###开始编译TensorFlow
```sh
[hyl@localhost tensorflow]$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: 
Please specify optimization flags to use during compilation [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? (Linux only) [Y/n] 
jemalloc enabled on Linux
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] 
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] 
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] 
No XLA support will be enabled for TensorFlow
Found possible Python library paths:
  /usr/lib/python2.7/site-packages
  /usr/lib64/python2.7/site-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python2.7/site-packages]

Using python library path: /usr/lib/python2.7/site-packages
Do you wish to build TensorFlow with OpenCL support? [y/N] 
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] 
No CUDA support will be enabled for TensorFlow
Configuration finished
.......
INFO: Starting clean (this may take a while). Consider using --expunge_async if the clean takes more than several minutes.
.......
INFO: All external dependencies fetched successfully.


```

在这里需要选择编译的各种选项，由于python2.7和python3.4可以共存，如果是需要用Python3来运行tensorflow，注意选择合适的python路径。如果前面安装了cuda和cudnn，在这里也可以选择支持cuda，一般使用缺省的路径即可。

开始编译tensorflow，生成pip package
```sh
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
...
...
Slow read: a 1891512136-byte read from /home/hyl/.cache/bazel/_bazel_hyl/b1fa4ed53787443b67fe0f9ecd924b96/execroot/tensorflow/bazel-out/local-dbg/bin/tensorflow/python/_pywrap_tensorflow.so took 13084ms.
Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 962.668s, Critical Path: 833.01s
```

需要注意的是：
	1. 需要编译debug版本的话，修改参数 -c opt为 -c dbg即可
	2. 需要支持cuda，加上参数--config=cuda
	3. 编译过程中需要访问google自己host的一些项目（boringssl），如果在墙内没办法编译，还是建议选择google编译好的版本。如果实在需要编译源代码，可以尝试修改bazel文件,避免依赖boringssl，应该也可以继续编译，不过我没有尝试。

生成whale安装包，注意最后生成的安装包的名字是和你的编译选项有关系的
```sh
[hyl@localhost tensorflow]$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
2017年 03月 13日 星期一 16:41:25 CST : === Using tmpdir: /tmp/tmp.Up8rhVg2Wa
~/repo/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles ~/repo/tensorflow
~/repo/tensorflow
/tmp/tmp.Up8rhVg2Wa ~/repo/tensorflow
2017年 03月 13日 星期一 16:41:26 CST : === Building wheel
~/repo/tensorflow
2017年 03月 13日 星期一 16:42:57 CST : === Output wheel file is in: /tmp/tensorflow_pkg
[hyl@localhost tensorflow]$ ls /tmp/tensorflow_pkg/
tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl
```
tensorflow-0.7.1-cp27-none-linux_x86_64.whl    -  python2.7  64位
tensorflow-0.7.1-cp34-cp34m-linux_x86_64.whl   -  python3.4  64位
tensorflow-0.7.1-py3-none-any.whl							 -  python3.4  64位 带debug信息

安装生成好的tensorflow包
```sh
[root@localhost hyl]# pip install /tmp/tensorflow_pkg/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl 
Processing /tmp/tensorflow_pkg/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl
Collecting six>=1.10.0 (from tensorflow==1.0.1)
  Downloading six-1.10.0-py2.py3-none-any.whl
Collecting protobuf>=3.1.0 (from tensorflow==1.0.1)
  Downloading protobuf-3.2.0-cp27-cp27mu-manylinux1_x86_64.whl (5.6MB)
    100% |████████████████████████████████| 5.6MB 295kB/s 
Requirement already satisfied: wheel in /usr/lib/python2.7/site-packages (from tensorflow==1.0.1)
Collecting mock>=2.0.0 (from tensorflow==1.0.1)
  Downloading mock-2.0.0-py2.py3-none-any.whl (56kB)
    100% |████████████████████████████████| 61kB 6.6MB/s 
Requirement already satisfied: numpy>=1.11.0 in /usr/lib64/python2.7/site-packages (from tensorflow==1.0.1)
Requirement already satisfied: setuptools in /usr/lib/python2.7/site-packages (from protobuf>=3.1.0->tensorflow==1.0.1)
Collecting funcsigs>=1; python_version < "3.3" (from mock>=2.0.0->tensorflow==1.0.1)
  Downloading funcsigs-1.0.2-py2.py3-none-any.whl
Collecting pbr>=0.11 (from mock>=2.0.0->tensorflow==1.0.1)
  Downloading pbr-2.0.0-py2.py3-none-any.whl (98kB)
    100% |████████████████████████████████| 102kB 6.1MB/s 
Installing collected packages: six, protobuf, funcsigs, pbr, mock, tensorflow
  Found existing installation: six 1.9.0
    Uninstalling six-1.9.0:
      Successfully uninstalled six-1.9.0
Successfully installed funcsigs-1.0.2 mock-2.0.0 pbr-2.0.0 protobuf-3.2.0 six-1.10.0 tensorflow-1.0.1
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

3. zipfile.LargeZipFile: Filesize would require ZIP64 extensions
    
    There is an issue reported here https://github.com/tensorflow/tensorflow/issues/5538, however, it's not tensorflow issue, one workround is to use --spawn_strategy=sandboxed
    
```sh
    bazel-bin/tensorflow/tools/pip_package/build_pip_package  --spawn_strategy=sandboxed /tmp/tf
```

4. ERROR: no such target '@local_config_cuda//crosstool:toolchain': target 'toolchain' not declared in package 'crosstool' defined by ......

   



