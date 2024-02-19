def create_lr_callbacks(start_val:int):
  return tf.keras.callbacks.LearningRateScheduler(lambda epoch: start_val * 10**(epoch/20))

