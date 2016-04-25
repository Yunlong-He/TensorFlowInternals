
## 一个简单的TensorFlow应用

TensorFlow主要是用来做深度学习的，不过从设计上来讲是支持一般的计算的，我们来看一个最简单例子，计算2+3=5

```python
import tensorflow as tf

# Create a variable.
a = tf.Variable(2)
b = tf.Variable(3)

c = tf.add(a, b)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(op)
result = sess.run(c)

print result
```

```sh
$ python test.py
```

运行之后最后输出5


和普通程序不同的是，TensorFlow并不是在语言层面直接计算2+3，而是需要调用session.run(...)，事实上真正的计算也是发生在这里。tensforflow是符号计算语言，前面几行只是定义了一个计算的流图，变量a,b,c分别是图的节点，session.run(op)用来初始化a, b，session.run(c)用来计算节点c，因为c依赖于a和b，所以在初始化a, b之前计算c是会报错的。

这里涉及到几个概念：
符号变量（Variable）：代表某个数据或某个操作
计算流图（Graph）：根据数据或者操作之间的依赖关系得到的一个图
会话（Session）：可以用来执行一个计算流图，并得到指定的变量结果

在没有显示声明计算流图的时候，tensorflow会缺省创建一个
```python
a = tf.Variable(2)
```
这一句会创建一个Tensor并加入到缺省的Graph里
最后加到variable list里的就是通过ops.get_default_graph().create_op(...)生成的Tensor

在查看create_op这个函数的实现时，我们可以看到调用了这样一个函数：
```python
def ops.convert_to_tensor：
#  This function converts Python objects of various types to `Tensor`
#  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
#  and Python scalars. For example:
```
这个函数非常有用，里面注册了tensorflow支持的各种可以转换的object类型及其相应的转换函数，比如对于常量2来说，这是一个python object，其对应的转换函数是_constant_tensor_conversion_function（python/ops/constant_op.py:175）

注意：在查看constant_op.py文件的时候，有这么几行用来注册转换类型及函数：
```python
ops.register_tensor_conversion_function(
    (list, tuple), _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.ndarray, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.generic, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200)
```

最后的数字100和200是检查转换类型的优先级，最generic的类型放在最后，所以object的优先级设为200，应该是最高的

注意：在加入到collection的时候，并没有像想象的那样用hashmap，而是直接放到一个列表里，可能是觉得变量不会太多，不值得用hashmap吧

```python
c = tf.add(a, b)
```
这里调用的是python/ops/gen_math_ops.py里的add函数，这个模块是编译时生成的。
```python
def add(x, y, name=None):
  return _op_def_lib.apply_op("Add", x=x, y=y, name=name)

_op_def_lib = _InitOpDefLibrary()

def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
```

看一下python/ops/op_def_library.py中函数的定义
```python
class OpDefLibrary(object):
  """Holds a collection of OpDefs, can add the corresponding Ops to a graph."""
```
说的很清楚，这个library包含了可以添加到Graph里的op的定义

在看函数apply_op之前，我们可以猜想一下这个函数应该做哪些事情：

	1. 根据op的名称查找对应的op的定义，这样我们可以检查用户的输入是不是正确
	2. 找到定义之后，我们得创建相应的op对象
	3. 把创建好的op对象加到graph里去 


两个需要注意的地方：一个是op是有参数的，要建立依赖关系；另一个是前面说过运行时可能有很多个graph，这个graph怎么选呢？很直接的想法就是看看它依赖哪个参数，参数在哪个graph里，就把它加到哪个graph里，如果依赖有两个参数，分别来自不同的graph，那我们应该把两个graph合并一下

看一下实现：
```python
    op_info = self._ops.get(op_type_name, None)

    g = ops._get_graph_from_inputs(_Flatten(keywords.values()))

      # Perform input type inference
      inferred_from = {}
      for input_arg in op_def.input_arg:

        if _IsListParameter(input_arg):
          ... 
       else:
          # In cases where we have an expected type, try to convert non-Tensor
          # arguments to that type.
```

再回过头来看刚才的代码：
_InitOpDefLibrary的作用是把相关的op注册到op library里，那这个操作是什么时候被调用的呢？通过pdb设置断点可以看到是初始化tensorflow这个模块的时候就调用了。
```gdb
(Pdb) b /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.py:3593
Breakpoint 1 at /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.py:3593
(Pdb) c
> /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.py(3593)<module>()
-> _op_def_lib = _InitOpDefLibrary()
(Pdb) bt
  /usr/lib/python2.7/bdb.py(400)run()
-> exec cmd in globals, locals
  <string>(1)<module>()
  /home/yunlong/repo/notes/tensorflow/examples/basic/arithmetic.py(2)<module>()
-> import tensorflow as tf
  /usr/local/lib/python2.7/dist-packages/tensorflow/__init__.py(23)<module>()
-> from tensorflow.python import *
  /usr/local/lib/python2.7/dist-packages/tensorflow/python/__init__.py(49)<module>()
-> from tensorflow import contrib
  /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/__init__.py(23)<module>()
-> from tensorflow.contrib import layers
  /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/layers/__init__.py(68)<module>()
-> from tensorflow.contrib.layers.python.layers import *
  /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/layers/python/layers/__init__.py(22)<module>()
-> from tensorflow.contrib.layers.python.layers.initializers import *
  /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/layers/python/layers/initializers.py(24)<module>()
-> from tensorflow.python.ops import random_ops
  /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/random_ops.py(27)<module>()
-> from tensorflow.python.ops import array_ops
  /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/array_ops.py(77)<module>()
-> from tensorflow.python.ops import gen_math_ops
> /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.py(3593)<module>()
-> _op_def_lib = _InitOpDefLibrary()
```

在执行语句result = sess.run(c)的时候，才会真正的计算a+b，那么我们想知道tensorflow里到底是怎么计算的。前面说过，session.run最终会调用到session.py里的这个函数
```python
 def _run(self, handle, fetches, feed_dict):

    unique_fetches, target_list, fetch_info = self._process_fetches(fetches)

    ret = []
    for fetch_names, fetch_contraction_fn in fetch_info:
      if fetch_names:
        fetched_vals = [fetched_results[name] for name in fetch_names]
        ret.append(fetch_contraction_fn(fetched_vals))
      else:
        ret.append(None)
```
计算过程似乎是调用了fetch_contraction_fn得到的，而这个函数是process_fetches的时候得到的。

<待续>

