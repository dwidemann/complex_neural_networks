# Complex Neural Networks
The purpose of this repo is build neural networks that take complex valued data, e.g. RF signal/digital communications, synthetic aperture radar (SAR) and process the data using complex operations. 


This code uses the complex tensor built William Falcon (below). Be careful to read his description of the tensor and what operations are supported. 


<p align="center">
<!--   <a href="https://williamfalcon.github.io/test-tube/">
    <img alt="react-router" src="https://raw.githubusercontent.com/williamfalcon/test-tube/master/imgs/test_tube_logo.png" width="50">
  </a> -->
</p>
<h2 align="center">
  Pytorch Complex Tensor
</h2>
<p align="center">
  Unofficial complex Tensor support for Pytorch 
</p>
<p align="center">
<a href="https://badge.fury.io/py/pytorch-complex-tensor"><img src="https://badge.fury.io/py/pytorch-complex-tensor.svg" alt="PyPI version" height="18"></a><!--   <a href="https://williamfalcon.github.io/test-tube/"><img src="https://readthedocs.org/projects/test-tube/badge/?version=latest"></a> -->
  <a href="https://github.com/williamFalcon/pytorch-complex-tensor/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
      <a href="https://circleci.com/gh/williamFalcon/pytorch-complex-tensor/"><img src="https://circleci.com/gh/williamFalcon/pytorch-complex-tensor.svg?style=svg"></a>

</p>   

### How it works

Treats first half of tensor as real, second as imaginary.  A few arithmetic operations are implemented to emulate complex arithmetic. Supports gradients.   

### Installation
```bash
pip install pytorch-complex-tensor
```

### Example:   
Easy import    

```python   
from pytorch_complex_tensor import ComplexTensor


### Supported ops:
| Operation | complex tensor | real tensor | complex scalar | real scalar |
| ----------| :-------------:|:-----------:|:--------------:|:-----------:|   
| addition | Y | Y | Y | Y |
| subtraction | Y | Y | Y | Y |
| multiply | Y | Y | Y | Y |
| mm | Y | Y | Y | Y |
| abs | Y | - | - | - |
| t | Y | - | - | - |
| grads | Y | Y | Y | Y |   