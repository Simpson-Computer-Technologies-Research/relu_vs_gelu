# ReLU vs. GeLU

There's definitely a pretty significant difference between the two (in the case of my example in `main.py`).

GeLU seems to be relatively more accurate than ReLU, but it's not always the case. It's always good to test out different activation functions to see which one works best for your specific use case.

### Example Output

An example output from running `main.py`

```
Test Loss (GeLU): 0.13099078834056854
Test Loss (ReLU): 0.31609782576560974
Actual: 30.0 | GeLU: 30.32281494140625 | ReLU: 30.15665626525879
Actual: 22.0 | GeLU: 21.602794647216797 | ReLU: 21.220478057861328
```
