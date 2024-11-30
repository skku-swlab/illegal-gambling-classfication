import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_single_image(model_path, image_path, target_size=(380, 380)):  # 크기를 380x380으로 수정
    """
    단일 이미지에 대한 예측을 수행하는 함수
    """
    # 모델 로드
    model = load_model(model_path)
    
    # 이미지 로드 및 전처리
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    
    # 예측 수행
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    # 결과 해석 (0.5를 임계값으로 사용)
    class_name = "illegal" if probability >= 0.5 else "normal"
    
    return class_name, probability

if __name__ == "__main__":
    # 경로 설정
    model_path = "./model/image_classification_model_b4.h5"  # 모델 파일 경로
    image_path = "./inference_test_images/normal1.png"  # 테스트할 이미지 경로
    
    # 예측 수행
    predicted_class, probability = predict_single_image(model_path, image_path)
    
    # 결과 출력
    print(f"\nPredicted class: {predicted_class}")
    print(f"Probability: {probability:.4f}")