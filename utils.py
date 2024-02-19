def create_lr_callbacks(start_val:int):
  """
  Description:
	Creates a learning rate scheduler callback for use in TensorFlow Keras models. This callback adjusts the learning rate dynamically during training based on a specified start value and a predefined function.

  Parameters:
  start_val (int): The initial learning rate value to be used at the beginning of training.

  Returns:
  A tf.keras.callbacks.LearningRateScheduler object.
  """
  return tf.keras.callbacks.LearningRateScheduler(lambda epoch: start_val * 10**(epoch/20))

