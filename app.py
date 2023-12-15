from flask import Flask, request, render_template, session
from flask_session import Session

import cv2
import glob
import numpy as np
import tensorflow as tf

app = Flask("유사 이미지 탐색 프로그램")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# 모델 로딩
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
model.trainable = False


# 이미지 처리 함수
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    features = model.predict(np.expand_dims(img, axis=0))
    return features.flatten()


# 이미지 유사성 평가 함수
def calculate_similarity(features1, features2):
    similarity = np.dot(features1, features2)
    max_similarity = np.dot(features1, features1)
    similarity_percentage = (similarity / max_similarity) * 100  # 유사도 백분율 변경
    return round(similarity_percentage, 2)


# 디렉토리에서 모든 이미지 파일 읽어오기
image_directory = 'static/images/'
pretrained_images = glob.glob(image_directory + '*.JPG')

# 사전에 읽어온 이미지들의 특징을 계산해서 미리 저장
pretrained_features = {image_path: extract_features(image_path) for image_path in pretrained_images}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 업로드된 이미지를 저장하고 특징 추출
        uploaded_file = request.files['file']
        uploaded_file.save('uploads/uploaded_image.jpg')
        uploaded_features = extract_features('uploads/uploaded_image.jpg')

        # 조회 카운트 세션으로 증가
        view_cnt = session.get('view_cnt', 0) + 1
        session['view_cnt'] = view_cnt

        # 유사도 계산
        similarity_scores = []
        for pretrained_image_path, pretrained_feature in pretrained_features.items():
            similarity = calculate_similarity(uploaded_features, pretrained_feature)
            similarity_scores.append((pretrained_image_path, similarity))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        return render_template('result.html', results=similarity_scores, view_cnt=view_cnt)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
