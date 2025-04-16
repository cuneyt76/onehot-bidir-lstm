# download and extract data, remove unnecessary folder
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
shutil.rmtree('/content/aclImdb/train/unsup')

# define datasets, define & adapt vectorizer using train inputs, define vectorized datasets
train_set = keras.utils.text_dataset_from_directory("aclImdb/train", validation_split=0.2, subset="training", seed=22, )
val_set   = keras.utils.text_dataset_from_directory("aclImdb/train", validation_split=0.2, subset="validation", seed=22, )
test_set  = keras.utils.text_dataset_from_directory("aclImdb/test", shuffle=False)

max_tokens, max_len = 25, 25
vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len, output_mode="int")
train_set_inputs = train_set.map(lambda x, y: x)
vectorizer.adapt(train_set_inputs)

train_set_ints = train_set.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_set_ints = val_set.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_set_ints = test_set.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# cache & prefetch for better performance
train_set_ints = train_set_ints.cache().prefetch(tf.data.AUTOTUNE)
val_set_ints = val_set_ints.cache().prefetch(tf.data.AUTOTUNE)
test_set_ints = test_set_ints.cache().prefetch(tf.data.AUTOTUNE)

# define a custom layer which produces onehot vector sequences from int sequences  
class OneHotLayer(layers.Layer):
  def __init__(self, max_tokens, **kwargs):
    super().__init__(**kwargs)
    self.max_tokens = max_tokens
  def call(self, inputs):
    return tf.one_hot(inputs, depth=self.max_tokens)
  def get_config(self):                                                # to ensure the 'depth' argument is saved correctly
    config = super().get_config()
    config.update({"max_tokens": self.max_tokens})
    return config

# define & compile model
inputs = keras.Input(shape=(None,), dtype="int64")
onehot_vector_sequences = OneHotLayer(max_tokens)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(onehot_vector_sequences)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# fit (train) model
callbacks = [keras.callbacks.ModelCheckpoint("onehot_bidir_lstm.keras", save_best_only=True)]
model.fit(train_set_ints, validation_data=val_set_ints, epochs=2, callbacks=callbacks)
model = keras.models.load_model("onehot_bidir_lstm.keras", custom_objects={"OneHotLayer": OneHotLayer})
print(f"Best model test acc: {model.evaluate(test_set_ints)[1]:.4f}")
