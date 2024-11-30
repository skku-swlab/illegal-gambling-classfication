import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# GPU 설정: MirroredStrategy 사용
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()  # GPU 병렬 처리 전략

# 데이터 경로 설정
dataset_dir = "./datasets"
img_size = (600, 600)  # EfficientNet-B7에 맞는 입력 크기
batch_size = 64
epochs = 15
learning_rate = 0.001
model_save_path = "./model/image_classification_model_b7.h5"

# 데이터 증강 및 로드
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    classes=["normal", "illegal"],
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    classes=["normal", "illegal"],
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# 모델 정의
with strategy.scope():
    base_model = EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "Precision", "Recall"])

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# 모델 저장
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

# 테스트 데이터 준비
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    classes=["normal", "illegal"],
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# 모델 평가
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

# 성능 지표 추가 분석
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = test_generator.classes
class_labels = ["normal", "illegal"]

roc_auc = roc_auc_score(true_classes, predictions)
f1 = f1_score(true_classes, predicted_classes)

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
print(f"\nROC-AUC Score: {roc_auc:.4f}, F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

# 그래프 그리기
plt.figure(figsize=(12, 4))

# Accuracy 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
