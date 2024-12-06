## Unit Tests

Unit tests are written to test the stability of some of the core modules in this project such as Tensor, Loss, and so on. We are using Google Test.

To run all the tests, run

```
cd tests
make
```

or `make test_xxx` if you want to test a specific module.


## Train with real dataset

Take Mnist for example. Run the following command to check the results.
```
cd example/Mnist
make
./mnist
```

There is also a python file in `python/` as a cross-reference. It's using the exactly the same network, loss function, optimizer, and initialization method for weights as in mnist.cc, so ideally, they should have very similar results.

#### SNNF

| Epoch | Batch 200 Loss | Batch 400 Loss | Batch 600 Loss | Batch 800 Loss |
|-------|----------------|----------------|----------------|----------------|
| 1     | 2.1857         | 2.12071        | 2.08646        | 2.0635         |
| 2     | 1.89894        | 1.88298        | 1.87258        | 1.86615        |
| 3     | 1.83421        | 1.83142        | 1.82995        | 1.82931        |
| 4     | 1.81792        | 1.8198         | 1.81547        | 1.81691        |
| 5     | 1.80539        | 1.79309        | 1.7807         | 1.7717         |
| 6     | 1.73855        | 1.73668        | 1.73593        | 1.73379        |

**Test Accuracy:** 74.6%

#### PyTorch

| Epoch | Batch 200 Loss | Batch 400 Loss | Batch 600 Loss | Batch 800 Loss |
|-------|----------------|----------------|----------------|----------------|
| 1     | 2.3017         | 2.2992         | 2.2961         | 2.2908         |
| 2     | 2.2326         | 2.2214         | 2.2103         | 2.1924         |
| 3     | 2.0262         | 2.0021         | 1.9761         | 1.9494         |
| 4     | 1.8029         | 1.7940         | 1.7858         | 1.7783         |
| 5     | 1.7436         | 1.7434         | 1.7416         | 1.7401         |
| 6     | 1.7349         | 1.7314         | 1.7286         | 1.7259         |

**Test Accuracy:** 75.84%

The model turns out to be reliable.


Another key aspect of measurement is **efficiency**. 
To be continued...