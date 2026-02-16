# DQN优化器集成指南

## 概述

本文档详细说明了基于DQN（Deep Q-Network）的BP/LTP协议参数优化系统，包括数据流匹配、奖励函数设计和DQN实现。

## 系统数据流验证

### 1. Sender → Optimizer 请求流程

#### Sender发送 (sender.py:181-241)
```python
request_data = {
    "data_size": int,                    # 待发送数据量
    "link_state": {                      # 链路状态
        "bit_error_rate": float,         # 误码率
        "delay_ms": float,               # 延时（毫秒）
        "transmission_rate_mbps": float  # 传输速率（Mbps）
    },
    "current_params": {                  # 当前协议参数
        "bundle_size": int,
        "ltp_block_size": int,
        "ltp_segment_size": int,
        "session_count": int
    },
    "timestamp": float                   # 时间戳
}
```

#### Optimizer接收 (mode_dqn.py:740-780)
```python
def handle_param_request(request_data):
    link_state = request_data.get("link_state", {})
    # 提取：data_size, bit_error_rate, delay_ms, transmission_rate_mbps

    # 使用DQN生成优化参数
    optimized_params = dqn_optimizer.optimize_params(link_state)

    return {
        "status": "success",
        "optimized_params": {            # 优化后的参数
            "bundle_size": int,
            "ltp_block_size": int,
            "ltp_segment_size": int,
            "session_count": int
        },
        "model_version": int,
        "model_info": dict,
        "timestamp": float
    }
```

✅ **接口匹配：完全对应，数据格式一致**

### 2. Sender → Receiver 元数据流程

#### Sender发送 (sender.py:370-403)
```python
metadata = {
    "type": "metadata",
    "data_size": int,                    # 数据量大小
    "link_state": {                      # 链路状态（完整）
        "bit_error_rate": float,
        "delay_ms": float,
        "transmission_rate_mbps": float
    },
    "protocol_params": {                 # 协议参数（完整）
        "bundle_size": int,
        "ltp_block_size": int,
        "ltp_segment_size": int,
        "session_count": int
    },
    "timestamp": float
}
```

#### Receiver接收 (receiver.py:278-347)
```python
def handle_metadata(data):
    data_size = metadata.get("data_size")
    link_state = metadata.get("link_state", {})
    protocol_params = metadata.get("protocol_params", {})

    # 计算交付时间
    delivery_time_ms = (end_timestamp - start_timestamp) * 1000

    # 生成训练记录
    record = {
        "input": {
            "data_size": data_size,
            "bit_error_rate": link_state["bit_error_rate"],
            "delay_ms": link_state["delay_ms"],
            "transmission_rate_mbps": link_state["transmission_rate_mbps"]
        },
        "output": {
            "bundle_size": protocol_params["bundle_size"],
            "ltp_block_size": protocol_params["ltp_block_size"],
            "ltp_segment_size": protocol_params["ltp_segment_size"],
            "session_count": protocol_params["session_count"]
        },
        "performance": {
            "delivery_time_ms": delivery_time_ms
        },
        "timestamp": time.time()
    }
```

✅ **接口匹配：完全对应，数据结构完整保留**

### 3. Receiver → Optimizer 训练数据流程

#### Receiver发送 (receiver.py:350-397)
```python
send_data = {
    "type": "training_records",
    "records": [                         # 批量记录列表
        {
            "input": {
                "data_size": int,
                "bit_error_rate": float,
                "delay_ms": float,
                "transmission_rate_mbps": float
            },
            "output": {
                "bundle_size": int,
                "ltp_block_size": int,
                "ltp_segment_size": int,
                "session_count": int
            },
            "performance": {
                "delivery_time_ms": float
            },
            "timestamp": float
        },
        ...  # 多条记录
    ],
    "count": int,                        # 记录数量
    "timestamp": float
}
```

#### Optimizer接收 (mode_dqn.py:810-843)
```python
def record_receive_server():
    message = json.loads(message_data)

    if message_type == "training_records":
        records = message.get("records", [])

        # 批量更新DQN模型
        dqn_optimizer.batch_update_model(records)
```

✅ **接口匹配：完全对应，记录格式与训练输入完美匹配**

## DQN实现详解

### 核心组件

#### 1. DQNNetwork (mode_dqn.py:64-202)

**架构**：
```
输入层 (4维状态)
    ↓
全连接层1 (128个神经元) + ReLU
    ↓
全连接层2 (128个神经元) + ReLU
    ↓
输出层 (16个Q值，对应16个动作)
```

**特点**：
- 主网络：用于选择动作和更新
- 目标网络：用于计算目标Q值，提供稳定训练
- 软更新：τ=0.001，逐步同步两个网络

**关键方法**：
```python
# 前向传播
q_values = network.forward(state, use_target=False)

# 反向传播（使用MSE损失）
loss = network.backward(state, q_targets)

# 更新目标网络（软更新）
network.update_target_network(tau=0.001)
```

#### 2. ExperienceReplay (mode_dqn.py:24-61)

**功能**：
- 存储经验: (state, action, reward, next_state, done)
- 随机采样：打破时间相关性
- 容量管理：最多10000条经验

**使用**：
```python
# 存储经验
replay.add((state, action, reward, next_state, done))

# 采样批次
batch = replay.sample_batch(batch_size=32)
```

#### 3. RewardCalculator (mode_dqn.py:204-319)

**多维度奖励函数**：

```
总奖励 = 0.5 × time_reward
       + 0.3 × throughput_reward
       + 0.2 × robustness_reward
```

##### a) 时间奖励 (权重0.5)

**目标**：最小化业务交付时间

**计算**：
```python
if delivery_time_ms ≤ min_time (100ms):
    time_reward = +1.0  # 最佳性能
elif delivery_time_ms ≥ max_time (5000ms):
    time_reward = -1.0  # 最差性能
else:
    # 线性插值
    time_ratio = (delivery_time - min_time) / (max_time - min_time)
    time_reward = 1.0 - 2.0 × time_ratio
```

**奖励曲线**：
```
  1.0 |•
      |  \
  0.0 |    \___
      |         \___
 -1.0 |              •
      +------------------
     100ms         5000ms
```

##### b) 吞吐量奖励 (权重0.3)

**目标**：最大化传输吞吐量

**计算**：
```python
throughput_mbps = (data_size × 8) / (delivery_time / 1000) / 1_000_000
throughput_normalized = min(throughput_mbps / 100.0, 1.0)
throughput_reward = 2.0 × throughput_normalized - 1.0
```

**含义**：
- 吞吐量达到100Mbps或以上：奖励 = +1.0
- 吞吐量为50Mbps：奖励 = 0.0
- 吞吐量接近0：奖励 = -1.0

##### c) 鲁棒性奖励 (权重0.2)

**目标**：在恶劣条件下仍能保持良好性能

**计算**：
```python
# 计算网络条件恶劣程度
ber_factor = min(bit_error_rate × 1e6, 1.0)
delay_factor = min(delay_ms / 500.0, 1.0)
adversity = (ber_factor + delay_factor) / 2.0

if adversity > 0.5:  # 恶劣条件
    if delivery_time_ms < max_time × 0.7:
        return +0.5  # 在恶劣条件下表现优秀
    elif delivery_time_ms < max_time:
        return 0.0
    else:
        return -0.5
else:  # 良好条件
    if delivery_time_ms < min_time × 2:
        return +0.3  # 在良好条件下也表现良好
    else:
        return 0.0
```

**意义**：
- 鼓励算法在高误码率或高延时环境下寻找稳健的解决方案
- 防止过度优化特定条件而在其他条件下表现糟糕

### DQN训练流程

#### 步骤1：接收训练记录

```python
def batch_update_model(records):
    for record in records:
        # 解包数据
        input_data = record["input"]
        output_data = record["output"]
        performance = record["performance"]

        # 构造状态
        state = {
            "data_size": input_data["data_size"],
            "bit_error_rate": input_data["bit_error_rate"],
            "delay_ms": input_data["delay_ms"],
            "transmission_rate_mbps": input_data["transmission_rate_mbps"]
        }
```

#### 步骤2：计算奖励

```python
        # 使用多维度奖励函数
        reward = reward_calculator.calculate_reward(
            delivery_time_ms=performance["delivery_time_ms"],
            data_size=input_data["data_size"],
            bit_error_rate=input_data["bit_error_rate"],
            delay_ms=input_data["delay_ms"]
        )
```

#### 步骤3：查找动作

```python
        # 根据输出参数反向查找动作
        action = _find_action_from_params(output_data)
```

#### 步骤4：存储经验

```python
        # 存入经验回放缓冲区
        store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,  # 简化处理
            done=False
        )
```

#### 步骤5：训练网络

```python
        # 如果有足够的经验，进行训练
        if len(experience_replay) >= batch_size:
            # 采样批次
            batch = experience_replay.sample_batch(batch_size)

            # 计算目标Q值
            for (s, a, r, s', done) in batch:
                if done:
                    target_q[a] = r
                else:
                    target_q[a] = r + γ × max(Q_target(s'))

            # 反向传播更新网络
            loss = network.backward(states, target_q)

            # 软更新目标网络
            network.update_target_network(τ=0.001)
```

#### 步骤6：衰减探索率

```python
        # 逐步减少探索
        epsilon = max(epsilon_min, epsilon × epsilon_decay)
        # epsilon_decay = 0.995
        # epsilon_min = 0.01
```

### DQN超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `state_dim` | 4 | 状态维度 |
| `action_dim` | 16 | 动作维度（4×4） |
| `hidden_dim` | 128 | 隐层神经元数 |
| `batch_size` | 32 | 批训练大小 |
| `learning_rate` | 0.001 | 学习率 |
| `gamma (γ)` | 0.99 | 折扣因子 |
| `epsilon (ε)` | 0.1 → 0.01 | 探索率（逐步衰减） |
| `epsilon_decay` | 0.995 | 探索率衰减系数 |
| `tau (τ)` | 0.001 | 目标网络软更新系数 |
| `replay_buffer_size` | 10000 | 经验回放缓冲区大小 |

## 状态空间和动作空间

### 状态空间（4维连续）

```
S = (data_size, bit_error_rate, delay_ms, transmission_rate_mbps)

归一化后：
s_normalized = [
    data_size / 100000,           # [0, 1]
    bit_error_rate × 1e6,         # [0, 1]
    delay_ms / 500,               # [0, 1]
    transmission_rate_mbps / 100  # [0, 1]
]
```

### 动作空间（16个离散动作）

```
A = {0, 1, 2, ..., 15}

映射关系：
action_idx = bundle_idx × 4 + block_idx

其中：
- bundle_idx ∈ {0, 1, 2, 3} → bundle_size ∈ {512, 1024, 2048, 4096}
- block_idx ∈ {0, 1, 2, 3} → ltp_block_size ∈ {256, 512, 1024, 2048}

同时：
- ltp_segment_size = ltp_segment_sizes[block_idx] ∈ {128, 256, 512, 1024}
- session_count = session_counts[bundle_idx] ∈ {1, 2, 4, 8}
```

**动作映射表**：

| action | bundle_size | ltp_block_size | ltp_segment_size | session_count |
|--------|-------------|----------------|------------------|---------------|
| 0      | 512         | 256            | 128              | 1             |
| 1      | 512         | 512            | 256              | 1             |
| 2      | 512         | 1024           | 512              | 1             |
| 3      | 512         | 2048           | 1024             | 1             |
| 4      | 1024        | 256            | 128              | 2             |
| 5      | 1024        | 512            | 256              | 2             |
| ...    | ...         | ...            | ...              | ...           |
| 15     | 4096        | 2048           | 1024             | 8             |

## DQN vs Q-Learning对比

| 特性 | Q-Learning (原mode.py) | DQN (mode_dqn.py) |
|------|------------------------|-------------------|
| **状态表示** | 离散化（10×10×10×10） | 连续（归一化向量） |
| **Q值存储** | Q表（字典） | 神经网络 |
| **泛化能力** | 弱（未见过的状态无法处理） | 强（可推广到未见过的状态） |
| **收敛速度** | 快（小状态空间） | 中等（需要更多样本） |
| **稳定性** | 较差（直接更新） | 好（目标网络+经验回放） |
| **奖励函数** | 简单（-时间） | 复杂（多维度加权） |
| **适用场景** | 简单环境 | 复杂连续状态空间 |

## 使用方法

### 启动DQN优化器

```bash
# 基本启动
python3 mode_dqn.py

# 自定义端口
python3 mode_dqn.py --param-port 5002 --record-port 5003
```

### 与原系统集成

1. **替换mode.py**：
```bash
cp mode_dqn.py mode.py
```

2. **或创建软链接**：
```bash
ln -sf mode_dqn.py mode.py
```

3. **无需修改sender.py和receiver.py**
   - 接口完全兼容
   - 数据格式一致

## 性能监控

### 模型信息输出

```python
model_info = {
    "model_version": int,      # 模型版本（更新次数）
    "training_steps": int,     # 训练步数
    "epsilon": float,          # 当前探索率
    "replay_buffer_size": int, # 经验回放缓冲区大小
    "avg_reward": float        # 最近100个episode的平均奖励
}
```

### 训练过程日志

```
[DQN训练] 开始使用 50 条记录进行批量训练
  [记录1/50] 奖励: 0.4523, 交付时间: 1234.56ms, Loss: 0.003421
  [记录11/50] 奖励: 0.3891, 交付时间: 1456.78ms, Loss: 0.002987
  ...
[DQN训练完成] 模型版本: 15, 平均奖励: 0.4201, 探索率ε: 0.0567, 训练步数: 480
```

### 参数请求日志

```
[新请求] 来自 192.168.1.1, 模型版本: 15
[参数请求] 数据量: 20480 bytes, 误码率: 0.0001, 延时: 150.0ms, 速率: 8.0Mbps
[参数响应] 已发送优化参数: {'bundle_size': 2048, 'ltp_block_size': 1024, ...}
```

## 优化建议

### 1. 调整奖励权重

根据具体需求调整三个维度的权重：

```python
# 更重视交付时间
total_reward = 0.7 * time_reward + 0.2 * throughput_reward + 0.1 * robustness_reward

# 更重视吞吐量
total_reward = 0.3 * time_reward + 0.6 * throughput_reward + 0.1 * robustness_reward

# 更重视鲁棒性
total_reward = 0.4 * time_reward + 0.2 * throughput_reward + 0.4 * robustness_reward
```

### 2. 调整超参数

```python
# 更激进的学习
learning_rate = 0.01  # 提高学习率
epsilon = 0.3         # 增加探索

# 更保守的学习
learning_rate = 0.0001  # 降低学习率
epsilon = 0.05          # 减少探索

# 更大的网络
hidden_dim = 256  # 增加隐层神经元
```

### 3. 增加网络层数

如果环境更复杂，可以增加网络深度：

```python
# 3层隐层网络
self.W1 = np.random.randn(state_dim, 256) * 0.01
self.W2 = np.random.randn(256, 128) * 0.01
self.W3 = np.random.randn(128, 64) * 0.01
self.W4 = np.random.randn(64, action_dim) * 0.01
```

## 总结

✅ **数据流验证**：
- Sender → Optimizer：完全匹配
- Sender → Receiver：完全匹配
- Receiver → Optimizer：完全匹配

✅ **DQN实现**：
- 深度神经网络替代Q表
- 经验回放提高样本效率
- 目标网络稳定训练过程

✅ **奖励函数**：
- 多维度设计（时间+吞吐量+鲁棒性）
- 加权组合平衡不同目标
- 归一化保证数值稳定

✅ **接口兼容**：
- 无需修改sender.py和receiver.py
- 直接替换mode.py即可使用

完整的DQN优化系统现已准备就绪！