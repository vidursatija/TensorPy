# TensorPy

TensorPy is a simple library to understand how automatic differentiation works within PyTorch. The classic numpy library for the same: [autograd](https://github.com/HIPS/autograd).

## Installation
```bash
git clone https://github.com/vidursatija/tensorpy
cd tensorpy
pip install -r requirements.txt
python3 setup.py install
```

## Usage
3 examples containing regression, classification and a simple neural network are given in the [examples](./examples) folder.

Simple differentiation usage:
```python
from tensorpy import Tensor
import numpy as np


a = Tensor(np.random.rand(2, 3), compute_grad=True)
b = Tensor(np.random.rand(3, 5), compute_grad=True)

c = a.matmul(b) # shape = (2, 5)
z = c.sum(axis=None) # shape = ()
print(z.value)
z.backward() # same as z.backward(np.ones_like(z.value))

print(a.grad) # dz / da = (dz / dc) * (dc / da)
print(b.grad) # dz / db = (dz / dc) * (dc / db)

```

## Contributing
This repo wasn't made to be used in production or development. It's just a simple library to see under the hood of ML libs like PyTorch. By no means is this library complete but I think it does a good job in explaining how gradients get calculated and propogated.

Even after saying this, pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](./LICENSE)