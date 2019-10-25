# Deep Structured Similarity Matching

### How to run it?
Example Usage:
```
python runscript.py --tanh_factors 1 --distance_parameter 4  --stride 2 --gamma_factor 0 --mult_factor 1 --NpSs 4
```

Please use `python runscript.py -h` to get info on what those parameters are.

### File Structure
```bash
├── mnist_data.py                       # fetch mnist data
├── runscript.py                        # run experiments
├── snn_multipleneurons_fast.py         # define deep structued sm class
└── README.md                           
```
