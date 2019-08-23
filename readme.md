# Complex Neural Networks
The purpose of this repo is build neural networks that take complex valued data, e.g. RF signal/digital communications, synthetic aperture radar (SAR) and process the data using complex operations. 


This code uses the complex tensor built William Falcon. Be careful to read his description of the tensor and what operations are supported. The readme is under **pytorch_complex_tensor**.

### TO RUN



* This notebook shows how to do operations and complex tensor. It's mostly a tutorial.
```python
cd notebooks
jupyter lab MetaClassWrapper.ipynb.
```
complex_convolution.ipynb notebook shows the basics for complex tensors. Ignore the code for complex convolution. It's wrong, I'll cut it later.  

### Note
I included the Falcon's complex tensor repo in ours because I thought we may need to modify it. If we don't need to change it, he has a pip install that may be better to use.
