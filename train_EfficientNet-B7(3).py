import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
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
batch_size = 128
epochs = 20
learning_rate = 0.001
model_save_path = "./model/image_classification_model_b7_improved.h5"
graph_save_dir = "./graphs"

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
    x = Dense(512, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # EfficientNet의 마지막 50개 레이어를 훈련 가능하게 설정
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "Precision", "Recall"])

# 콜백 설정
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    batch_size=batch_size,
    class_weight={0: 2., 1: 1.},  # 'normal' 클래스에 더 높은 가중치 부여
    callbacks=[early_stopping, reduce_lr]
)

# 학습 종료 시간 기록
end_time = time.time()

# 학습 시간 출력
training_time = end_time - start_time
print(f"Training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")

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

# 테스트 데이터 평가 및 지표 추가
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = test_generator.classes
class_labels = ["normal", "illegal"]

# 혼동 행렬
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# F1 Score, Precision, Recall, ROC-AUC Score
roc_auc = roc_auc_score(true_classes, predictions)
f1 = f1_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 학습 기록 그래프 저장
os.makedirs(graph_save_dir, exist_ok=True)

# 정확도와 손실 그래프
plt.figure(figsize=(12, 6))

# 정확도 변화
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()

# 손실 변화
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()

# 그래프 저장
accuracy_loss_graph_path = os.path.join(graph_save_dir, "accuracy_loss_graph.png")
plt.tight_layout()
plt.savefig(accuracy_loss_graph_path)
plt.close()  # 그래프를 띄우지 않고 저장만 함

print(f"Accuracy and Loss graph saved to: {accuracy_loss_graph_path}")