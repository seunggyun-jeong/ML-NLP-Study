# 오토인코더 신경망 구축 알고리즘의 이해

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 학습을 위한 설정 값 지정
learning_rate = 0.02 # 최적화를 위한 하이퍼 파라미터
training_epochs = 50 # 학습 횟수 (전체 Batch를 한 번 둘러 보는 것을 1 epoch이라고 함)
                # 셔플링을 통해 데이터를 무작위로 섞어 학습에 효과적
batch_size = 256 # Mini-batch의 사이즈
display_step = 1 # 손실함수의 출력 주기
examples_to_show = 10 # 보여줄 MNIST Reconstruction 이미지 개수

# 신경망 구조 정의
input_size = 784 # input은 28*28
hidden1_size = 256
hidden2_size = 128

def downloadData():
    # MNIST 데이터를 다운로드 합니다.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 이미지들을 float32 데이터 타입으로 변경합니다.
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    # 28*28 형태의 이미지를 784차원으로 flattening 합니다.
    x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
    # [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
    x_train, x_test = x_train / 255., x_test / 255.

    return (x_train, y_train), (x_test, y_test)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져온다.
def getBatch(x_train):
    # data를 mini batch로 분리
    train_data = tf.data.Dataset.from_tensor_slices(x_train)
    train_data = train_data.shuffle(60000).batch(batch_size)
    return train_data

def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# Autoencoder 모델 정의
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 인코딩(Encoding) - 784 -> 256 -> 128
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        # 디코딩(Decoding) 128 -> 256 -> 784
        self.hidden_layer_3 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())                                                    
        self.output_layer = tf.keras.layers.Dense(input_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        H2_output = self.hidden_layer_2(H1_output)
        H3_output = self.hidden_layer_3(H2_output)
        reconstructed_x = self.output_layer(H3_output)

        return reconstructed_x

# MSE 손실 함수 정의
@tf.function
def mse_loss(y_pred, y_true):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))

optimizer = tf.optimizers.RMSprop(learning_rate)

@tf.function
def train_step(model, x):
    y_true = x
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y_true)
    gredients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gredients, model.trainable_variables))

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = downloadData()
    train_data = getBatch(x_train)
    
    # AutoEncoder 모델 정의
    AutoEncoder_model = AutoEncoder()

    # 최적화 수행
    for epoch in range(training_epochs):
        # Autoencoder는 Unsupervised Learning이므로 타겟 레이블 y가 필요하지 않음
        for batch_x in train_data:
            # 옵티마이저 실행
            _, current_loss = train_step(AutoEncoder_model, batch_x), mse_loss(AutoEncoder_model(batch_x), batch_x)

        # 학습결과 출력
        if epoch % display_step == 0:
            print(f"반복 : {epoch + 1}번째, 현재 손실 : {current_loss}")

    # 테스트 데이터로 Reconstruction을 수행
    reconstructed_result = AutoEncoder_model(x_test[:examples_to_show])
    # 원본 MNIST 데이터와 Reconstruction 결과를 비교합니다.
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(x_test[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))
    f.savefig('reconstructed_mnist_image.png')  # reconstruction 결과를 png로 저장합니다.
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
