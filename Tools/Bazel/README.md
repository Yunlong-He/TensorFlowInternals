
**关于bazel**

bazel是TensorFlow的构建工具，相当于cmake或者maven，据说google内部的项目都用bazel来构建。主要目的是用简单的方法解决项目之间的依赖关系。

类似于maven会把下载下来的包存放在在$HOME下的.m2目录里，bazel会把下载的文件放在$HOME/.cache/bazel下，但是有个不同，不同的编译会在下面产生不同的目录。参考https://bazel.build/versions/master/docs/output_directories.html。 

目录的结构是：$HOME/.cache/bazel/_bazel_$USER/md5($WORKSPACE_PATH)。由此可见，workspace的path决定了cache的路径。

也许这种放置文件的方式可以保证独立性，但是后果也很明显，比如把tensorflow源代码移动到另外一个地方重新编译，原来的文件就得重新下载一遍，因为要放到不同的目录下，动辄重新下载十几G的文件，对我这种国内用户来说代价太大了。

事实上，bazel提供了重新设定cache路径的方法，就是传递参数--output_base，但是这样也不是很灵活，比如我有两个tensorflow目录，源代码有些不同，想共用一个cache目录存放下载的文件，可以么？

**bazel的设计目标**
Requirements for an output directory layout:

1. 支持多用户，不会相互冲突
2. 支持同时编译多个workspace
3. 对于同一个workspace，支持编译多个目标配置
4. 不和其他工具冲突
5. 访问简单
6. 易于清理，即使是选择性的清理其中一部分
7. 无二义性，即使遇到用户使用符号链接等情况
8. 每个用户的编译状态都被放置到同一个目录下


**bazel的目录结构**
```sh
<workspace-name>/                         <== The workspace directory
  bazel-my-project => <...my-project>     <== Symlink to execRoot
  bazel-out => <...bin>                   <== Convenience symlink to outputPath
  bazel-bin => <...bin>                   <== Convenience symlink to most recent written bin dir $(BINDIR)
  bazel-genfiles => <...genfiles>         <== Convenience symlink to most recent written genfiles dir $(GENDIR)

/home/user/.cache/bazel/                  <== Root for all Bazel output on a machine: outputRoot
  _bazel_$USER/                           <== Top level directory for a given user depends on the user name:
                                              outputUserRoot
    install/
      fba9a2c87ee9589d72889caf082f1029/   <== Hash of the Bazel install manifest: installBase
        _embedded_binaries/               <== Contains binaries and scripts unpacked from the data section of
                                              the bazel executable on first run (e.g. helper scripts and the
                                              main Java file BazelServer_deploy.jar)
    7ffd56a6e4cb724ea575aba15733d113/     <== Hash of the client's workspace directory (e.g.
                                              /home/some-user/src/my-project): outputBase
      action_cache/                       <== Action cache directory hierarchy
                                              This contains the persistent record of the file metadata
                                              (timestamps, and perhaps eventually also MD5 sums) used by the
                                              FilesystemValueChecker.
      action_outs/                        <== Action output directory. This contains a file with the
                                              stdout/stderr for every action from the most recent bazel run
                                              that produced output.
      command.log                         <== A copy of the stdout/stderr output from the most recent bazel
                                              command.
      external/                           <== The directory that remote repositories are downloaded/symlinked
                                              into.
      server/                             <== The Bazel server puts all server-related files (such as socket
                                              file, logs, etc) here.
        server.log                        <== Server logs.
      <workspace-name>/             <== Working tree for the Bazel build & root of symlink forest: execRoot
        _bin/                             <== Helper tools are linked from or copied to here.

        bazel-out/                        <== All actual output of the build is under here: outputPath
          local_linux-fastbuild/          <== one subdirectory per unique target BuildConfiguration instance;
                                              this is currently encoded
            bin/                          <== Bazel outputs binaries for target configuration here: $(BINDIR)
              foo/bar/_objs/baz/          <== Object files for a cc_* rule named //foo/bar:baz
                foo/bar/baz1.o            <== Object files from source //foo/bar:baz1.cc
                other_package/other.o     <== Object files from source //other_package:other.cc
              foo/bar/baz                 <== foo/bar/baz might be the artifact generated by a cc_binary named
                                              //foo/bar:baz
              foo/bar/baz.runfiles/       <== The runfiles symlink farm for the //foo/bar:baz executable.
                MANIFEST
                <workspace-name>/
                  ...
            genfiles/                     <== Bazel puts generated source for the target configuration here:
                                              $(GENDIR)
              foo/bar.h                       e.g. foo/bar.h might be a headerfile generated by //foo:bargen
            testlogs/                     <== Bazel internal test runner puts test log files here
              foo/bartest.log                 e.g. foo/bar.log might be an output of the //foo:bartest test with
              foo/bartest.status              foo/bartest.status containing exit status of the test (e.g.
                                              PASSED or FAILED (Exit 1), etc)
            include/                      <== a tree with include symlinks, generated as needed.  The
                                              bazel-include symlinks point to here. This is used for
                                              linkstamp stuff, etc.
          host/                           <== BuildConfiguration for build host (user's workstation), for
                                              building prerequisite tools, that will be used in later stages
                                              of the build (ex: Protocol Compiler)
        <packages>/                       <== Packages referenced in the build appear as if under a regular workspace
```

**bazel的常用命令**

bazel clean

**其他**
如果有时间，我会创建几个例子来说明如何使用bazel