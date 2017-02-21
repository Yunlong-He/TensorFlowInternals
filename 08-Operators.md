
这是一个自定义op的例子，
REGISTER_OP("Fact")
    .Output("fact: string")
    .Doc(R"doc(
Output a fact about factorials.
)doc");

class FactOp : public OpKernel {
 public:
  explicit FactOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Output a scalar string.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_tensor));
    auto output = output_tensor->template scalar<string>();

    output() = "0! == 1";
  }
};

REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_CPU), FactOp);

运行的时候可以这样调用：
a = tf.user_ops._fact()
sess = tf.Session()
b = sess.run(a)
print(b)

输出结果：b"Chuck Norris is Jeff Dean's 20% project."

代码实现里最重要的两条是
    auto output = output_tensor->template scalar<string>();
    output() = "0! == 1";

output的类型是TTypes<T>::Scalar，该定义是这样的：
typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
      Eigen::Aligned> Scalar;

关于Eigen::TensorMap

TensorMap是Eigen包unsupported中的一个文件，只有特定版本的eigen才有，比如这里http://eigen.tuxfamily.org/dox-devel/unsupported/group__CXX11__Tensor__Module.html, 或者用谷歌tensorflow里的版本
/home/yunlong/.cache/bazel/_bazel_yunlong/f6d234c2641c3ae91dbbe2e4e22d8f75/external/eigen_archive/eigen-eigen-c5e90d9e764e/

所谓的unsupported，是指由第三方（非eigen官方）提供的api，eigen不提供支持。

关于TensorMap的实现，可以查看文件：
/home/yunlong/.cache/bazel/_bazel_yunlong/f6d234c2641c3ae91dbbe2e4e22d8f75/external/eigen_archive/eigen-eigen-c5e90d9e764e/unsupported/Eigen/CXX11/src/Tensor/TensorMap.h

TensorMap是一个类，其中对于operator()的无参数实现为：
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar& operator()() const
    {
      EIGEN_STATIC_ASSERT(NumIndices == 0, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return m_data[0];
    }

  private:
    Scalar* m_data;
    Dimensions m_dimensions;

Scalar的定义:
template<typename PlainObjectType, int Options_> class TensorMap : public TensorBase<TensorMap<PlainObjectType, Options_> >
{
  public:
    typedef typename internal::traits<PlainObjectType>::Scalar Scalar;


下面我们看看Op是怎样注册的：

#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                       \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr \
      TF_ATTRIBUTE_UNUSED = ::tensorflow::OpDefBuilder(name)

这里是先创建OpDefBuilder，然后根据它创建OpDefBuilderReceiver.

在op_def_builder.h里，OpDefBuilder类里有两个函数Input和Ouput，在相关的注释里给出了对应字符串的要求
  // Adds an input or output to this OpDefBuilder (and returns *this).
  // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
  // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
  // * For a single tensor: <type>
  // * For a sequence of tensors with the same type: <number>*<type>
  // * For a sequence of tensors with different types: <type-list>
  // Where:
  //   <type> is either one of "float", "int32", "string", ...
  //                 or the name of an attr (see above) with type "type".
  //   <number> is the name of an attr with type "int".
  //   <type-list> is the name of an attr with type "list(type)".
  // TODO(josh11b): Indicate Ref() via an optional argument instead of
  // in the spec?
  // TODO(josh11b): SparseInput() and SparseOutput() matching the Python
  // handling?
  OpDefBuilder& Input(StringPiece spec);
  OpDefBuilder& Output(StringPiece spec);

如果输入的字符串不符合规则，比如Input("inputBits: int32")，看起来没问题，但是不符合全部小写的规则，就会报下面的错误，但是这个错误在文档上是没有的，有些坑爹了
yunlong@dl-y9:~/github/tf/tf-py3$ bazel build -c dbg --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package
INFO: Found 1 target...
ERROR: /home/yunlong/github/tf/tf-py3/tensorflow/python/BUILD:684:1: Executing genrule //tensorflow/python:user_ops_pygenrule failed: namespace-sandbox failed: error executing command
  (cd /home/yunlong/.cache/bazel/_bazel_yunlong/f6d234c2641c3ae91dbbe2e4e22d8f75/tf-py3 && \
  exec env - \
    PATH=/home/yunlong/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-7.5/bin:/home/yunlong/bin \
  /home/yunlong/.cache/bazel/_bazel_yunlong/f6d234c2641c3ae91dbbe2e4e22d8f75/tf-py3/_bin/namespace-sandbox @/home/yunlong/.cache/bazel/_bazel_yunlong/f6d234c2641c3ae91dbbe2e4e22d8f75/tf-py3/bazel-sandbox/13c60e1e-ee31-4fba-94a8-d69c1aa59f1c-0.params -- /bin/bash -c 'source external/bazel_tools/tools/genrule/genrule-setup.sh; bazel-out/host/bin/tensorflow/python/gen_user_ops_py_wrappers_cc Fact 0 > bazel-out/local_linux-py3-dbg/genfiles/tensorflow/python/ops/gen_user_ops.py').
F tensorflow/core/framework/op.cc:119] Check failed: ::tensorflow::Status::OK() == (RegisterAlreadyLocked(op_def)) (OK vs. Invalid argument: Missing type for input ''; in OpDef: name: "FlipBits" input_arg { } output_arg { name: "flipbits" type: DT_INT32 } summary: "Output a integer with flipped bits")Attempting to register: Op<name=FlipBits; signature=: -> flipbits:int32>
/bin/bash: line 1: 44705 Aborted                 (core dumped) bazel-out/host/bin/tensorflow/python/gen_user_ops_py_wrappers_cc Fact 0 > bazel-out/local_linux-py3-dbg/genfiles/tensorflow/python/ops/gen_user_ops.py
Target //tensorflow/tools/pip_package:build_pip_package failed to build

OpDefBuilderReceiver的构造函数是这样的：
OpDefBuilderReceiver::OpDefBuilderReceiver(const OpDefBuilder& builder) {
  OpDef op_def;
  builder.Finalize(&op_def);
  OpRegistry::Global()->Register(op_def);
}



下面是一个Add操作的例子，支持不同类型的输入，以及不同的device，有点复杂

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    auto To = output->flat<T>();

    const int num = ctx->num_inputs();
    if (num == 1) {
      *output = input0;
      return;
    }
#define I(IDX) ctx->input(IDX).flat<T>()

#if defined(__ANDROID_TYPES_SLIM__)
    // On Android by default,we only support additions of two arguments, so we
    // can reduce the number of template instantiations.
    OP_REQUIRES(ctx, num == 2,
                errors::InvalidArgument("Only additions of two arguments "
                                        "supported. Num inputs: ",
                                        num));
    functor::Add2Functor<Device, T> functor2;
    functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
#else
    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(ctx->template eigen_device<Device>(), To, I(r), I(r + 1),
                I(r + 2), I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
#endif  // defined(__ANDROID_TYPES_SLIM__)

#undef I
  }
};

#define REGISTER_ADDN(type, dev)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AddN").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      AddNOp<dev##Device, type>)

#define REGISTER_ADDN_CPU(type) REGISTER_ADDN(type, CPU)

TF_CALL_NUMBER_TYPES(REGISTER_ADDN_CPU);
#undef REGISTER_ADDN_CPU

#if GOOGLE_CUDA
REGISTER_ADDN(float, GPU);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("inputs")
                            .HostMemory("sum"),
                        AddNOp<CPUDevice, int32>);
#endif  // GOOGLE_CUDA

#undef REGISTER_ADDN

}  // namespace tensorflow
