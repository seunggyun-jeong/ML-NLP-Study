# 텐서플로우를 이용한 ANN 구현

import tensorflow as tf

# 학습을 위한 설정 값 지정
learning_rate = 0.001 # 최적화를 위한 하이퍼 파라미터
num_epochs = 30 # 학습 횟수 (전체 Batch를 한 번 둘러 보는 것을 1 epoch이라고 함)
                # 셔플링을 통해 데이터를 무작위로 섞어 학습에 효과적
batch_size = 256 # Mini-batch의 사이즈
display_step = 1 # 손실함수의 출력 주기
# ANN의 구조 정의
input_size = 784 # input은 28*28
hidden1_size = 256
hidden2_size = 256
output_size = 10

def downloadData():
    # MNIST 데이터를 다운로드 합니다.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 이미지들을 float32 데이터 타입으로 변경합니다.
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    # 28*28 형태의 이미지를 784차원으로 flattening 합니다.
    x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
    # [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
    x_train, x_test = x_train / 255., x_test / 255.
    # 레이블 데이터에 one-hot encoding을 적용합니다.
    y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

    return (x_train, y_train), (x_test, y_test)

def getBatch(x_train, y_train):
    # data를 mini batch로 분리
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.shuffle(60000).batch(batch_size)
    return train_data

# weight, bias 초기화를 위한 메서드
# 랜덤으로 초기 weight, bias 값을 선정할 수 있음
def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# keras Model을 통해 ANN 모델 정의
class ANN(tf.keras.Model):
    def __init__(self):
        super(ANN, self).__init__() # tf.keras.Model의 생성자 상속

        # fully connected layer 구현
        # kernel = weight
        # bias = bias

        # 1번 hidden layer
        # activation function == ReLU
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='relu',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        # 2번 hidden layer
        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
                                                    activation='relu',
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())
        # output layer                                                    
        self.output_layer = tf.keras.layers.Dense(output_size,
                                                    activation=None,
                                                    kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                    bias_initializer=random_normal_initializer_with_stddev_1())                                                    

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        H2_output = self.hidden_layer_2(H1_output)
        logits = self.output_layer(H2_output)

        return logits

# cross-entropy 손실 함수 정의
# api를 사용하여 softmax function과 Cross-Entropy 계산을 한 번에 할 수 있음
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

if __name__ == '__main__':
    print("ch5 prac-----------")

    # MNIST Data Download
    # ch4와 같은 과정
    (x_train, y_train), (x_test, y_test) = downloadData()

    # mini batch로 분리
    train_data = getBatch(x_train=x_train, y_train=y_train)

    # 최적화를 위한 Adam 옵티마이저 정의
    # tf.optimizers 참고
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    optimizer = tf.optimizers.Adam(learning_rate)

    # ANN 모델 정의
    ANN_model = ANN()

    # 지정된 횟수만큼 최적화 수행
    for epoch in range(num_epochs):
        average_loss = 0.
        total_batch = int(x_train.shape[0] / batch_size)

        # 모든 batch들에 대해 최적화 수행
        for batch_x, batch_y in train_data:
            # 옵티마이저를 실행하여 파라미터 업데이트
            _, current_loss = train_step(ANN_model, batch_x, batch_y), cross_entropy_loss(ANN_model(batch_x), batch_y)
            # 평균 손실 측정
            average_loss += current_loss / total_batch
            
        #지정된 epoch마다 학습결과 출력
        if epoch % display_step == 0:
            print(f"반복(Epoch): {epoch+1}, 손실 함수(Loss): {average_loss}")
        
    print(f"정확도 == {compute_accuracy(ANN_model(x_test), y_test)}")