import tensorflow as tf

# TensorFlow 버전 확인
print("TensorFlow version:", tf.__version__)

# GPU 인식 확인
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# CUDA 사용 가능 여부 확인
print("CUDA Built:", tf.test.is_built_with_cuda())
