
**关于TensorFlow的性能**

刚刚做了一个测试，使用docker hub上的镜像tensorflow/tensorflow:1.0.0-devel-gpu，在在E5-2640上运行TensorFlow-Examples里的logistic regression时，会报出如下的警告信息：
```sh
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
```

检查源代码我们会发现cpu_feature_guard.h里声明了一个函数WarnAboutUnusedCPUFeatures()，从名字里可以看出tensorflow在运行时会检查cpu是否有更好的feature可以被利用到。
```c++
namespace tensorflow {
namespace port {

// Called by the framework when we expect heavy CPU computation and we want to
// be sure that the code has been compiled to run optimally on the current
// hardware. The first time it's called it will run lightweight checks of
// available SIMD acceleration features and log warnings about any that aren't
// used.
void WarnAboutUnusedCPUFeatures();

}  // namespace port
}  // namespace tensorflow
```
在cpu_feature_guard.cc文件里还有另一个函数CheckFeatureOrDie()，从名字也可以看出，当前编译的tensorflow如果使用了某些CPU的feature，但是在实际运行的机器上如果不存在该feature的话，tensorflow会拒绝执行。，这样避免了实际运行中出现各种奇怪的兼容性问题。
