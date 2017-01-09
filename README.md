# The Predictron

A TensorFlow implementation of
```
The Predictron: End-To-End Learning and Planning
David Silver, Hado van Hasselt, Matteo Hessel, Tom Schaul, Arthur Guez, et al.
```
[arXiv](https://arxiv.org/abs/1612.08810)

## Run
Try it with
```
python ./train_multigpu.py --max_depths=<2,4,8,16> --batch_size=128 \
  --num_gpus=<your available number of GPUs>
```

We assume `batch_size` can be divided by `num_gpus`.

## Requirements:
tensorflow=0.12.1

## Some Results
`max_step = 16` and Training on `8` NVIDIA GTX TITAN X GPUs. It takes quite a long time to achieve the same numbers of steps reported in the paper. Here are some results during the training procedure.
```
INFO:multigpu_train:2017-01-09 01:11:36.704713: step 159760, loss = 0.0043, loss_preturns = 0.0018, loss_lambda_preturns = 0.0004 (2679.7 examples/sec; 0.048 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:39.854633: step 159770, loss = 0.0038, loss_preturns = 0.0017, loss_lambda_preturns = 0.0002 (2888.7 examples/sec; 0.044 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:43.026452: step 159780, loss = 0.0067, loss_preturns = 0.0031, loss_lambda_preturns = 0.0002 (2848.1 examples/sec; 0.045 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:46.252385: step 159790, loss = 0.0099, loss_preturns = 0.0035, loss_lambda_preturns = 0.0014 (3272.3 examples/sec; 0.039 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:49.477405: step 159800, loss = 0.0032, loss_preturns = 0.0013, loss_lambda_preturns = 0.0003 (3051.7 examples/sec; 0.042 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:52.570256: step 159810, loss = 0.0046, loss_preturns = 0.0020, loss_lambda_preturns = 0.0003 (3314.3 examples/sec; 0.039 sec/batch)
INFO:multigpu_train:2017-01-09 01:11:55.710512: step 159820, loss = 0.0040, loss_preturns = 0.0017, loss_lambda_preturns = 0.0003 (3374.0 examples/sec; 0.038 sec/batch)
```
