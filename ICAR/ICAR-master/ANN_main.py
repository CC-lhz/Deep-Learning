# FileName:ANN_main.py
# coding = utf-8
# Created by Hzq
from ANN_utils import *
from CAR_utils import *
from time import sleep
import datetime



# 函数 s_1()
# 1.生成训练数据：
# 通过np.linspace生成一个包含6个点的等差数列，表示在0到2π之间的6个点。
# 创建一个大小为(6, 2)的数组train_y，其中第一列存储对应点的sin值，第二列存储cos值。

# 2.构建神经网络模型：
# 使用 ANN_utils 中的 MlpNn 类创建了一个多层感知器（MLP）神经网络模型。
# 通过调用 model.add_layer 方法，添加了四个隐藏层和一个输出层，每个隐藏层都使用tanh激活函数，输出层使用linear激活函数。
# 通过 model.compile 编译模型，使用Adam优化器。

# 3.训练神经网络：
# 使用生成的训练数据进行模型训练，进行1000次epoch。
# 学习速率为0.001。
# 打印训练信息。

# 4.模型预测：
# 对模型进行单一输入（np.pi/3）的预测，并打印结果。


def s_1():
    train_x = np.linspace(0, 2*np.pi, 6)
    train_y = np.zeros((train_x.shape[0], 2))
    train_y[:, 0] = np.sin(train_x)
    train_y[:, 1] = np.cos(train_x)

    model = MlpNn(input_shape=1)
    model.add_layer(32, 'tanh')
    model.add_layer(32, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(2, 'linear')
    model.compile(optimizer='adam')
    for i in range(1000):
        model.fit(train_x, train_y, epoch=1, learn_rate=0.001, verbose=1)
    a = np.mat([np.pi/3])
    print(model.predict(a), a.shape)




# 函数 s_4()
# 1.初始化环境和代理：
# 创建一个汽车模拟环境 (EnvCar) 和一个深度Q网络代理 (DQNAgent)。
# 设置状态空间大小 (state_size) 和动作空间大小 (action_size)。
# 2.执行剧集：
# 对于每一个剧集（episode），重置环境并获取初始状态。
# 在每个时间步内，代理采取动作，观察下一个状态和奖励。
# 将经验存储到代理的记忆中。
# 3.训练代理：
# 如果满足一定条件，使用经验回放 (agent.replay()) 对代理进行训练。
# 训练是通过从记忆中随机抽样一批经验，然后利用这批经验来更新Q网络参数。
#4. 打印信息：
# 打印每个剧集的信息，包括得分、探索率 (epsilon) 和记忆长度。
# 每5个剧集保存一次代理模型。
def s_4():
    env = EnvCar()

    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    batch_size = 16
    EPISODES = 1000
    max_time = 2500

    for e in range(EPISODES):
        state = env.reset()[:, 0]
        state = np.reshape(state, [1, state_size])

        for time in range(max_time):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(delay=1, display=1, action=action, mode='map', generation=e, verbose=1)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            if done or time == max_time-1:
                print("episode: {}/{}, score: {}, e: {:.2}, len: {}"
                      .format(e, EPISODES, time, agent.epsilon, len(agent.memory)))
                if e % 5 == 0:
                    agent.save()
                break
            # start_time = datetime.datetime.now()
            if len(agent.memory) > batch_size and time % 2 == 0:
                agent.replay(batch_size)
            # end_time = datetime.datetime.now()
            # print(len(agent.memory), end_time - start_time)


s_4()

