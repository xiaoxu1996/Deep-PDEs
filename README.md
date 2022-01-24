# Deep-PDEs
Our code is included in 3 files, Allen.py, BSB.py, HJB.py. \
Some special attentions, we will explain below.
## Our configuration
python == 3.8\
tensorflow-gpu == 2.4.0\
numpy == 1.19.5
## Different dimensions
We can change the dimension directly by setting **d**.\
## Neural network
We define the multiscale fusion function "_lnn_time_net", and the convolutional network function "_cnn_time_net". 
If you choose multiscale fusion, use "_lnn_time_net" when updating **A** and **Gamma**; 
if you choose convolutional neural network, use "_cnn_time_net" when updating **A** and **Gamma**.
## Update the network every time
We use **reuse=tf.compat.v1.AUTO_REUSE** to update network.
## Optimization
For Allen-Cahn equation, we use **tf.compat.v1.train.GradientDescentOptimizer**.\
For other equation, we use **tf.compat.v1.train.AdamOptimizer**.
## Learning rate
There are two types of learning rate updates, **tf.compat.v1.train.exponential_decay**, **tf.compat.v1.train.piecewise_constant**.
## Extra
In convolutional neural network, there is a **reshape** operation on the input and a **reshape** operation in the last linear layer.
