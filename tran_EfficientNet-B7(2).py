import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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
epochs = 10
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

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    batch_size=batch_size,
    class_weight={0: 1., 1: 2.}  # class_weight 설정으로 정상 사이트의 정확도를 높임
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
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
print(f"\nROC-AUC Score: {roc_auc:.4f}, F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

# ROC Curve
fpr, tpr, _ = roc_curve(true_classes, predictions)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_value)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(true_classes, predictions)
avg_precision = average_precision_score(true_classes, predictions)

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, color='green', lw=2, label='Precision-Recall curve (AP = %0.2f)' % avg_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
