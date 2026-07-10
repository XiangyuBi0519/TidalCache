[root@mep-mirror-280t-ga-az5-turbo-86 test_gather]# export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 
[root@mep-mirror-280t-ga-az5-turbo-86 test_gather]# export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
[root@mep-mirror-280t-ga-az5-turbo-86 test_gather]# bash build_and_test.sh
=== Environment ===
ASCEND_HOME_PATH: /usr/local/Ascend/cann-8.5.1
Python: Python 3.11.14

=== Checking prerequisites ===
torch: 2.9.0+cpu
torch_npu: 2.9.0.post1+gitee7ba04
Custom op header: FOUND
Custom op library: FOUND

=== Building wrapper ===
    self.build_extensions()
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/utils/cpp_extension.py", line 1082, in build_extensions
    build_ext.build_extensions(self)
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/setuptools/_distutils/command/build_ext.py", line 484, in build_extensions
    self._build_extensions_serial()
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/setuptools/_distutils/command/build_ext.py", line 510, in _build_extensions_serial
    self.build_extension(ext)
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/setuptools/command/build_ext.py", line 264, in build_extension
    _build_ext.build_extension(self, ext)
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/setuptools/_distutils/command/build_ext.py", line 565, in build_extension
    objects = self.compiler.compile(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/utils/cpp_extension.py", line 866, in unix_wrap_ninja_compile
    _write_ninja_file_and_compile_objects(
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/utils/cpp_extension.py", line 2223, in _write_ninja_file_and_compile_objects
    _run_ninja_build(
  File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/utils/cpp_extension.py", line 2614, in _run_ninja_build
    raise RuntimeError(message) from e
RuntimeError: Error compiling objects for extension
[ERROR] 2026-07-10-15:05:00 (PID:2306, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception

=== Running tests ===
ERROR: gather_kv_wrapper not found. Build first with:
  python setup.py build_ext --inplace
