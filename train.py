import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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
normal_dir = os.path.join(dataset_dir, "normal")  # 정상 사이트
illegal_dir = os.path.join(dataset_dir, "illegal")  # 불법 도박 사이트

# 하이퍼파라미터 설정
img_size = (260, 260)  # EfficientNet-B2에 맞는 입력 크기
batch_size = 32
epochs = 10
learning_rate = 0.001
model_save_path = "./model/image_classification_model.h5"  # 저장 경로

# 데이터 증강 및 로드
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1  # 데이터셋의 10%를 검증 데이터로 사용
)

train_generator = datagen.flow_from_directory(
    dataset_dir,            # normal, illegal 디렉터리가 포함된 기본 디렉터리
    classes=["normal", "illegal"],  # 명시적으로 클래스 지정
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    classes=["normal", "illegal"],  # 명시적으로 클래스 지정
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# 모델 정의: MirroredStrategy로 감싸기
with strategy.scope():
    # ImageNet으로 전이 학습된 EfficientNet-B2 모델 로드
    base_model = EfficientNetB2(
        include_top=False,  # 마지막 레이어 제외
        weights="imagenet",  # ImageNet 사전 학습 가중치
        input_shape=(img_size[0], img_size[1], 3)
    )

    # 모델 구성
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Base model 고정 (Fine-tuning용)
    for layer in base_model.layers:
        layer.trainable = False

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# 모델 저장
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 디렉터리 생성
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# 테스트 데이터 준비
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    classes=["normal", "illegal"],  # 명시적으로 클래스 지정
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 성능 지표 출력
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

true_classes = test_generator.classes
class_labels = ["normal", "illegal"]  # 클래스 이름 명시

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))
