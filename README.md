# ReLU vs. GeLU

There's definitely a pretty significant difference between the two (in the case of my example in `main.py`).

GeLU seems to be relatively more accurate than ReLU, but it's not always the case. It's always good to test out different activation functions to see which one works best for your specific use case.

### Example Output

An example output from running `main.py`

```
Test Loss (GeLU): 0.2649143636226654
Test Loss (ReLU): 0.591556191444397
Actual: 30.0 | GeLU: 30.477054595947266 | ReLU: 30.39864730834961
Actual: 22.0 | GeLU: 21.806781768798828 | ReLU: 21.342248916625977
```
