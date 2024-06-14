# ReLU vs. GeLU

There's definitely a pretty significant difference between the two (in the case of my example in `main.py`).

GeLU seems to out-perform ReLU.

### Example Output

An example output from running `main.py`

```
Test Loss (GeLU): 0.08334237337112427
Test Loss (ReLU): 0.45898932218551636

GeLU Network:
tensor([[29.8971],
        [21.5473]], grad_fn=<AddmmBackward0>)
ReLU Network:
tensor([[30.1095],
        [20.9291]], grad_fn=<AddmmBackward0>)
```
