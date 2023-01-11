import tensorflow as tf

# Data Load & Preprocessing
def loadData():
    # numpy array type으로 반환
    # int 형 데이터 타입
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # float32 데이터 타입으로 변환
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

    # 28 * 28 형태의 이미지를 784차원으로 flattening
    # -1은 현재 데이터에 넘버를 자동으로 맞춰주는 Magic Number
    x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

    # [0, 255] 사이의 값을 [0, 1] 사이의 값으로 Nomalize
    x_train, x_test = x_train / 255., x_test / 255.

    # 레이블 데이터에 one-hot encoding을 적용
    # 정답이 0 ~ 9까지 총 10개의 정답 경우의 수가 있으므로 depth를 10으로 둠
    # 0 == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # 1 == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # ...
    y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

    return (x_train, y_train), (x_test, y_test)

# batch
def batching(x_train, y_train):
    # tensor dataset 형태로 반환
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Mini-Batch 생성
    # 100개 씩 묶은 mini-Batch 생성
    train_data = train_data.repeat().shuffle(60000).batch(100)
    return iter(train_data)

# keras subclassing Softmax Regression 모델 정의
# tf.keras.Model을 상속받음
class SoftmaxRegression(tf.keras.Model):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        # a(Wx + b) 형태의 연산을 정의한 Dense API 활용
        # activation = a
        # kernel = W
        # bias = b
        # 784차원의 flatten된 10Dimension 의 softmaxRegression 모델으로 전환
        self.softmax_layer = tf.keras.layers.Dense(10,
                                                    activation=None,
                                                    kernel_initializer='zeros',
                                                    bias_initializer='zeros')

    def call(self, x):
        logits = self.softmax_layer(x)
        return tf.nn.softmax(logits)

# 손실 함수 정의
@tf.function
def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=[1]))

# 그래디언트 옵티마이저 정의
optimizer = tf.optimizers.SGD(0.5)

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

if __name__ == "__main__" :
    print("hello")
    (x_train, y_train), (x_test, y_test) = loadData()
    train_data_iter = batching(x_train, y_train)
    SoftmaxRegression_model = SoftmaxRegression()

    # 1000번 반복을 통해 최적화 수행
    for i in range(1000):
        batch_xs, batch_ys = next(train_data_iter)
        train_step(SoftmaxRegression_model, batch_xs, batch_ys)

    print(f"정확도 == {compute_accuracy(SoftmaxRegression_model(x_test), y_test)}")