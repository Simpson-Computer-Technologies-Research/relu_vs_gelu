# ReLU vs. GeLU

There's definitely a pretty significant difference between the two (in the case of my example in `main.py`).

GeLU seems to be relatively more accurate than ReLU, but it's not always the case. It's always good to test out different activation functions to see which one works best for your specific use case.

### Example Output

An example output from running `main.py`

```
Test Loss (GeLU): 0.21306896209716797
Test Loss (ReLU): 0.6141756772994995
Actual: 30.0 | GeLU: 30.40980339050293 | ReLU: 30.139482498168945
Actual: 22.0 | GeLU: 21.787561416625977 | ReLU: 21.228818893432617
```
