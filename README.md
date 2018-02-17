# tf-troubleshooting
Tips and tricks for troubleshooting TF 

## Test `input_fn` only
Give a dummy `model_fn`
```python
def model_fn(features, labels, mode, params, config):
    train_op = tf.train.RMSPropOptimizer(learning_rate=params.learning_rate).minimize(
        loss=tf.get_variable('test', initializer=tf.constant(1.0)), global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=tf.constant(1.0),
        train_op=train_op)
```

Then 

```python
model = tf.estimator.Estimator(model_fn=nade.model_fn, params=params)
while True:
    model.train(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN), steps=params.eval_step)
```

## Check what `input_fn` outputs
```python
sess.run(input_data.input_fn(ModeKeys.TRAIN)[0])
```

## exit code 139 (interrupted by signal 11: SIGSEGV)
No other information, just throw this error and exit.

Reason: `tf.nn.embedding_lookup` is out of boundary.

One can create `dummy_loss = tf.reduce_sum(Xc_embd)` and now you see where the error comes from.

## Speed comparision: 1.5.0 vs. 1.6.0rc0 build from source
1.5.0:
```log
INFO:tensorflow:global_step/sec: 1.03983
INFO:tensorflow:loss = 5.2569094, step = 101 (96.172 sec)
INFO:tensorflow:global_step/sec: 1.25702
INFO:tensorflow:loss = 5.145889, step = 201 (79.551 sec)
INFO:tensorflow:global_step/sec: 1.06523
INFO:tensorflow:loss = 4.992849, step = 301 (93.876 sec)
```
1.6.0rc0
```log
INFO:tensorflow:loss = 5.260799, step = 1
INFO:tensorflow:global_step/sec: 1.36537
INFO:tensorflow:loss = 5.2473655, step = 101 (73.240 sec)
INFO:tensorflow:global_step/sec: 1.38253
INFO:tensorflow:loss = 5.1599884, step = 201 (72.339 sec)
```
20% speedup? may not really worth it since building tf takes couple of hours.

## Hanging when running tf
```log
2018-02-14 20:43:11.179530: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0, compute capability: 3.7)
```
One time thing when running TF for first time, wait. or just do `sudo reboot`.

## Get certain time from [B,T,D] tensor
```python
    import numpy as np

    B = 1
    T = 10
    D = 128
    a = tf.constant(np.random.random([B, T, D]))
    x = tf.constant([[1, 2, 3]])
    sequence_length = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(x))
        u = tf.tile(tf.expand_dims(tf.range(0, B), 1), [1, 3])
        s = tf.stack([u, x], axis=2)
        print(sess.run(s))

        print(sess.run(tf.shape(tf.gather_nd(a, s))))

    exit()
```
