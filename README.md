# The Predictron

A TensorFlow implementation of  
```
The Predictron: End-To-End Learning and Planning  
David Silver, Hado van Hasselt, Matteo Hessel, Tom Schaul, Arthur Guez, et al. 
```
[arXiv](https://arxiv.org/abs/1612.08810)

Try it with
```
# We assume batch_size can be divided by num_gpus
python ./train_multigpu.py --max_depths=<2,4,8,16> --batch_size=128 \
  --num_gpus=<your available number of GPUs>
```
