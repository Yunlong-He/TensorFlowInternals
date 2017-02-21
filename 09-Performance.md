
**1.0的改变**
在r0.10的版本里，很多op的实现是分cpu版本和gpu版本的，运行的时候根据device来选择合适的op kernel。
比如cast这个operator:

```sh
REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);
````
但是在1.0里，变成了如下写法：

```sh
class CastOp : public XlaOpKernel {
 public:
  explicit CastOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &dst_dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(src_dtype_, &src_type_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dst_dtype_, &dst_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* builder = ctx->builder();
    xla::ComputationDataHandle input = ctx->Input(0);
    xla::ComputationDataHandle output;

    if (src_dtype_ == dst_dtype_) {
      output = input;
    } else if (src_dtype_ == DT_BOOL) {
      // XLA's ConvertElementType doesn't support casting to/from
      // bools. So we need to handle those cases separately.
      // Builds the equivalent of (input ? 1 : 0)
      xla::ComputationBuilder l(builder->client(), "PredCast");
      xla::ComputationDataHandle x =
          l.Parameter(0, xla::ShapeUtil::MakeShape(src_type_, {}), "x");
      l.Select(x, XlaHelpers::One(&l, dst_dtype_),
               XlaHelpers::Zero(&l, dst_dtype_));
      xla::Computation computation = l.Build().ConsumeValueOrDie();
      output = builder->Map({input}, computation);
    } else if (dst_dtype_ == DT_BOOL) {
      output = builder->Ne(input, XlaHelpers::Zero(builder, src_dtype_));
    } else {
      output = builder->ConvertElementType(input, dst_type_);
    }

    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(CastOp);
};

REGISTER_XLA_OP("Cast", CastOp);
````
我还没有仔细研究，不过大概猜测试使用compiler的上下文信息对cast的实现进行选择，把对应的实现插入到graph里。

**关于移植及兼容性**

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
