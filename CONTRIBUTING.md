Contributing to libgrape-lite
=============================

libgrape-lite has been developed by an active team of software engineers and
researchers. We encourage the open-source community contribute to us to improve
this project.

libgrape-lite is licensed under [Apache License 2.0][2].

Newcomers to libgrape-lite
--------------------------

For newcomers to libgrape-lite, you could find instructions about how to build
and run applications using libgrape-lite in [README][3].

libgrape-lite is hosted on Github, and use Github issues as the bug tracker.
you can [file an issue][4] when you meets trouble when working with libgrape-lite.

Before creating a new bug entry, we recommand you first [search][7] among existing
libgrape-lite bugs to see if it has already been resolved.

When creating a new bug entry, please provide necessary information of your
problem in the description , such as operating system version, libgrape-lite
version, and other system configurations to help us diagnose the problem.

We also welcome any help on libgrape-lite from the community, including but not
limited to fixing bugs and adding new features. Note that you need to sign
the [CLA][5] before submmit patches to us.

Contributing a patch
--------------------

### Working conversions

libgrape-lite follows the [Google C++ Style Guide][1]. When submitting patches
to libgrape-lite, please first format your code with clang-format by
the Makefile command `make clformat`, and make sure your code doesn't break the
cpplint convension using the CMakefile command `make cpplint`.

#### Open a pull request

Contributing to libgrape-lite requires submitting your patch as a Github pull
request. We'll ask you to prefix the pull request title with the issue number
and the kind of patch (`BUGFIX` or `FEATURE`) in brackets, for example,
`[BUGFIX-1234] Fix crash in running SSSP on the p2p graph` or
`[FEATURE-2345] Support loading graph data from HDFS`.

libgrape-lite runs typical graph analytical algorithms on the `p2p-31` graph
as the test suite. Before submitting your pull request, make sure the test suite
hasn't been break. You can run the test suite using the following command under
the `build` directory,

```bash
../misc/app_tests.sh
```

#### Resolve conflict

You generally do NOT need to rebase your pull requests unless there are merge
conflicts with master. When Github complaining that "Can’t automatically merge"
on your pull request, you'll be asked to rebase your pull request on top of
the latest master branch, using the following commands:

+ First rebasing to the most recent master:

      git remote add upstream https://github.com/alibaba/libgrape-lite.git
      git fetch upstream
      git rebase upstream/master

+ Then git may show you some conflicts when it cannot merge, say `conflict.cpp`,
  you need
  - Maually modify the file to resolve the conflicts
  - After resolved, mark it as resolved by
  
        git add conflict.cpp

+ Then you can continue rebase by

      git rebase --continue

+ Finally push to your fork, then the pull request will be got updated:

      git push --force

Contributing to the documentation
---------------------------------

libgrape-lite provides comprehensive documents to explain the underlying
design and implementation details. The documentation follows the syntax
of Doxygen markup. If you find anything you can help, submit pull request
to us and thanks for your enthusiasm!

[1]: https://google.github.io/styleguide/cppguide.html
[2]: https://github.com/alibaba/libgrape-lite/blob/master/LICENSE
[3]: https://github.com/alibaba/libgrape-lite/blob/master/README.md
[4]: https://github.com/alibaba/libgrape-lite/issues/new/choose
[5]: https://cla-assistant.io/alibaba/libgrape-lite
[6]: https://alibaba.github.io/libgrape-lite
[7]: https://github.com/alibaba/libgrape-lite/pulls
