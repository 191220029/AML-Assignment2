# AML-Assignment2

## Getting Started
Dependacy  requires the C++ PyTorch library (libtorch) in version v2.2.0 to be available on your system. You can either:

- Use the system-wide libtorch installation (default).
Install libtorch manually and let the build script know about it via the LIBTORCH environment variable.
- Use a Python PyTorch install, to do this set LIBTORCH_USE_PYTORCH=1.
When a system-wide libtorch can't be found and LIBTORCH is not set, the build script can download a pre-built binary version of libtorch by using the download-libtorch feature. By default a CPU version is used. The TORCH_CUDA_VERSION environment variable can be set to cu117 in order to get a pre-built binary using CUDA 11.7.
- System-wide Libtorch
On linux platforms, the build script will look for a system-wide libtorch library in /usr/lib/libtorch.so.

### Python PyTorch Install
If the `LIBTORCH_USE_PYTORCH` environment variable is set, the active python interpreter is called to retrieve information about the torch python package. This version is then linked against.

### Libtorch Manual Install
- Get `libtorch` from the PyTorch website download section and extract the content of the zip file.
- For Linux and macOS users, add the following to your `.bashrc` or equivalent, where `/path/to/libtorch` is the path to the directory that was created when unzipping the file.
```
export LIBTORCH=/path/to/libtorch
```
The header files location can also be specified separately from the shared library via the following:
```
# LIBTORCH_INCLUDE must contain `include` directory.
export LIBTORCH_INCLUDE=/path/to/libtorch/
# LIBTORCH_LIB must contain `lib` directory.
export LIBTORCH_LIB=/path/to/libtorch/
```
- For Windows users, assuming that `X:\path\to\libtorch` is the unzipped libtorch directory.

 - - Navigate to Control Panel -> View advanced system settings -> Environment variables.
Create the `LIBTORCH` variable and set it to `X:\path\to\libtorch`.
- - Append `X:\path\to\libtorch\lib` to the Path variable.
If you prefer to temporarily set environment variables, in PowerShell you can run
```
$Env:LIBTORCH = "X:\path\to\libtorch"
$Env:Path += ";X:\path\to\libtorch\lib"
```