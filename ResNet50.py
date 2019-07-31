import keras
keras.utils.plot_model(keras.applications.ResNet50(include_top=True,input_shape=(224,224,3),weights=None), to_file='ResNet50.png', show_shapes=True)
