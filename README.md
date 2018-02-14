# tf-troubleshooting
Quick tricks for troubleshooting TF 

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
