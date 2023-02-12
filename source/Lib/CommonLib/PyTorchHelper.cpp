//
// Created by ywlee on 2/12/2023.
//

#include "PyTorchHelper.h"

PyTorchHelper::PyTorchHelper(string fpath, bool useGpu, int nGpu)
{
  try
  {
    ifstream ifs(fpath, std::ifstream::binary);

    if (torch::cuda::is_available() && useGpu)
    {
      device = torch::Device(format_string("cuda:%d", nGpu));
    }

    model = torch::jit::load(ifs, device);
    model.eval();
  }
  catch (const c10::Error &)
  {
    std::cerr << " Error raised while loading model : " << fpath << "\n";
  }
}