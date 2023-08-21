//
// Created by ywlee on 2/12/2023.
//

#ifndef NEXTSOFTWARE_PYTORCHHELPER_H
#define NEXTSOFTWARE_PYTORCHHELPER_H

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

class PyTorchHelper
{
public:
  torch::jit::script::Module model;
  torch::Device              device = torch::kCPU;
  torch::NoGradGuard         no_grad;

public:
  PyTorchHelper(string fpath, int nGpu);

  template<typename... Args> string format_string(const string &format, Args... args)
  {
    size_t             size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
    unique_ptr<char[]> buffer(new char[size]);
    snprintf(buffer.get(), size, format.c_str(), args...);
    return string(buffer.get(), buffer.get() + size - 1);
  }
};

#endif   // NEXTSOFTWARE_PYTORCHHELPER_H
