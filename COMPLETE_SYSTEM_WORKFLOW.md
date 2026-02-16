# BP/LTP自适应优化系统 - 完整工作流程文档

## 📋 文档概述

本文档详细说明了BP/LTP自适应优化系统的**完整13步工作流程**，包括三个节点（发送节点A、接收节点B、算力单元C）之间的交互，以及当前系统实现与流程的对应关系。

---

## 🏗️ 系统架构概览

### 三个主要组件

```
┌─────────────────────────────────────────────────────────────────┐
│                     系统架构总览                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  节点A (发送)              节点B (接收)         算力单元C (优化)  │
│  ┌──────────────┐        ┌──────────────┐    ┌──────────────┐  │
│  │              │        │              │    │              │  │
│  │ sender.py    │◄──────►│ receiver.py  │◄──►│ mode_dqn_v2  │  │
│  │              │        │              │    │   .py        │  │
│  └──────────────┘        └──────────────┘    └──────────────┘  │
│        │                       │                      ▲          │
│        │CSV配置                │记录器                │          │
│        ▼                       ▼                      │          │
│   network_config.csv   training_records.log   训练数据反馈  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 通信端口

| 组件 | 端口 | 用途 |
|------|------|------|
| Sender → Receiver | 5001 | 传输元数据、完成通知 |
| Sender → Computer C | 5002 | 参数请求、参数响应 |
| Receiver → Computer C | 5003 | 训练记录上传 |
| BP/LTP协议 | 其他 | 实际数据传输 |

---

## 🔄 完整13步工作流程

### 第1步：发送节点读取配置并收集链路状态

**执行位置**: `sender.py` → `get_link_state()` 方法

**流程**:
```python
# 步骤1.1：从CSV文件读取业务配置
- 读取CSV配置: data_size, BER, delay, transmission_rate
- 获取当前链路状态

# 步骤1.2：收集当前协议栈参数
protocol_params = {
    "bundle_size": 1024,
    "ltp_block_size": 512,
    "ltp_segment_size": 256,
    "session_count": 1
}
```

**代码位置**:
- [sender.py:90-150](/root/agent/send/sender.py#L90-L150) - `load_config_from_csv()` 和 `get_link_state()`

**输出**:
```json
{
  "data_size": 10240,
  "bit_error_rate": 1e-5,
  "delay_ms": 100.0,
  "transmission_rate_mbps": 10.0
}
```

---

### 第2步：发送节点A向电脑C请求优化参数

**执行位置**: `sender.py` → `request_optimization_params()` 方法

**流程**:
```
Sender (A)                           Computer C
    │                                    │
    ├─────── TCP 5002 ────────────────►│
    │   JSON请求:                       │
    │   {                               │
    │     "model_version": 5,           │
    │     "data_size": 10240,           │
    │     "link_state": {               │
    │       "bit_error_rate": 1e-5,    │
    │       "delay_ms": 100.0,         │
    │       "transmission_rate_mbps": 10│
    │     }                             │
    │   }                               │
    │                                   │◄─处理中
    │◄─────── TCP 5002 ───────────────┤
    │   JSON响应:                       │
    │   {                               │
    │     "bundle_size": 2048,          │
    │     "ltp_block_size": 1024,      │
    │     "ltp_segment_size": 512,     │
    │     "session_count": 3            │
    │   }                               │
    │                                   │
```

**代码位置**:
- [sender.py](/root/agent/send/sender.py) - `request_optimization_params()` 方法
- [mode_dqn_v2.py:490-502](/root/agent/computer/mode_dqn_v2.py#L490-L502) - `optimize_params()` 方法

**关键数据**:
- **请求格式**: 包含模型版本、数据大小、链路状态
- **响应格式**: 4个优化后的协议参数
- **通信协议**: TCP JSON消息

---

### 第3步：发送节点A修改协议栈参数并配置链路

**执行位置**: `sender.py` → `apply_protocol_params()` 方法

**流程**:
```python
# 步骤3.1：更新本地协议参数
self.protocol_params = {
    "bundle_size": 2048,
    "ltp_block_size": 1024,
    "ltp_segment_size": 512,
    "session_count": 3
}

# 步骤3.2：应用到BP/LTP协议栈
if self.use_bp_ltp:
    self.bp_ltp_interface.setup_parameters(
        bundle_size=2048,
        ltp_block_size=1024,
        ltp_segment_size=512,
        session_count=3
    )

# 步骤3.3：配置链路参数
self.bp_ltp_interface.configure_link(
    delay=100.0,  # 毫秒
    bit_error_rate=1e-5,
    transmission_rate_mbps=10.0
)
```

**代码位置**:
- [sender.py](/root/agent/send/sender.py) - `apply_protocol_params()` 方法
- [bp_ltp_interface.py](/root/agent/send/bp_ltp_interface.py) - BP/LTP接口实现

---

### 第4步：发送节点A发送数据大小和Bundle大小给节点B

**执行位置**: `sender.py` → `notify_receiver_prepare()` 方法

**流程**:
```
Sender (A)                           Receiver (B)
    │                                    │
    ├─────── TCP 5001 ────────────────►│
    │   JSON元数据:                      │
    │   {                                │
    │     "data_size": 10240,           │
    │     "bundle_size": 2048,          │
    │     "ltp_block_size": 1024,      │
    │     "ltp_segment_size": 512,     │
    │     "session_count": 3,           │
    │     "timestamp": 1234567890       │
    │   }                                │
    │                                   │◄─计算bundle数
    │◄─────── TCP 5001 ───────────────┤
    │   确认消息:                        │
    │   {"status": "ready"}             │
    │                                   │
```

**代码位置**:
- [sender.py](/root/agent/send/sender.py) - `notify_receiver_prepare()` 方法
- [receiver.py](/root/agent/receive/receiver.py) - `receive_metadata()` 方法

**关键数据**:
```python
metadata = {
    "data_size": data_size,
    "bundle_size": bundle_size,
    "ltp_block_size": ltp_block_size,
    "ltp_segment_size": ltp_segment_size,
    "session_count": session_count,
    "timestamp": time.time()
}
```

---

### 第5步：接收节点B计算Bundle数量并开启BP/LTP接收

**执行位置**: `receiver.py` → `prepare_to_receive()` 方法

**流程**:
```python
# 步骤5.1：计算需要接收的bundle数量
num_bundles = math.ceil(data_size / bundle_size)
# 例如: 10240 / 2048 = 5个bundle

# 步骤5.2：开启BP/LTP接收流程
self.bp_ltp_receiver.start_receiving(
    expected_bundles=num_bundles,
    bundle_size=bundle_size,
    ltp_block_size=ltp_block_size,
    ltp_segment_size=ltp_segment_size,
    session_count=session_count
)

# 步骤5.3：记录接收开始时间
receive_start_time = time.time()
```

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `prepare_to_receive()` 方法

---

### 第6步：发送节点A记录时间t1并开始传输

**执行位置**: `sender.py` → `send_data()` 方法

**流程**:
```python
# 步骤6.1：记录传输开始时间
transmission_start_time = time.time()  # t1

# 步骤6.2：在BP/LTP协议下发送数据
self.bp_ltp_interface.send_bundles(
    data=payload,
    num_bundles=num_bundles,
    bundle_size=bundle_size,
    ltp_block_size=ltp_block_size,
    session_count=session_count
)

# 步骤6.3：等待传输完成
transmission_end_time = time.time()  # t1'
```

**代码位置**:
- [sender.py](/root/agent/send/sender.py) - `send_data()` 方法
- [bp_ltp_interface.py](/root/agent/send/bp_ltp_interface.py) - 实际发送实现

**关键变量**:
```python
t1 = transmission_start_time  # 用于步骤9中发送给接收端
```

---

### 第7步：接收节点B接收完毕并记录时间t2

**执行位置**: `receiver.py` → `wait_for_completion()` 方法

**流程**:
```python
# 步骤7.1：等待所有数据接收完毕
received_data = self.bp_ltp_receiver.wait_for_completion()

# 步骤7.2：记录接收完成时间
transmission_complete_time = time.time()  # t2

# 步骤7.3：计算业务交付时间
delivery_time = transmission_complete_time - t1_from_sender  # 等待步骤9
```

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `wait_for_completion()` 方法

---

### 第8步：接收节点B通知发送节点A传输完毕

**执行位置**: `receiver.py` → `notify_sender_complete()` 方法

**流程**:
```
Receiver (B)                        Sender (A)
    │                                    │
    ├─────── TCP 5001 ────────────────►│
    │   JSON消息:                       │
    │   {                               │
    │     "status": "complete",        │
    │     "received_bytes": 10240,     │
    │     "completion_time": 1.234     │
    │   }                               │
    │                                   │
```

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `notify_sender_complete()` 方法
- [sender.py](/root/agent/send/sender.py) - 接收完成通知处理

---

### 第9步：发送节点A发送传输信息给接收节点B

**执行位置**: `sender.py` → `send_transmission_info()` 方法

**流程**:
```
Sender (A)                          Receiver (B)
    │                                    │
    ├─────── TCP 5001 ────────────────►│
    │   JSON信息:                       │
    │   {                               │
    │     "transmission_start_time": t1,│
    │     "data_size": 10240,          │
    │     "bit_error_rate": 1e-5,     │
    │     "delay_ms": 100.0,           │
    │     "transmission_rate_mbps": 10, │
    │     "bundle_size": 2048,         │
    │     "ltp_block_size": 1024,     │
    │     "ltp_segment_size": 512,    │
    │     "session_count": 3,          │
    │     "model_version": 5           │
    │   }                               │
    │                                   │◄─构建记录
    │                                   │
```

**代码位置**:
- [sender.py](/root/agent/send/sender.py) - `send_transmission_info()` 方法
- [receiver.py](/root/agent/receive/receiver.py) - 接收并处理

**关键数据**:
```python
transmission_info = {
    "transmission_start_time": t1,  # 步骤6记录
    "data_size": data_size,          # 步骤1读取
    "bit_error_rate": ber,           # 步骤1读取
    "delay_ms": delay_ms,            # 步骤1读取
    "transmission_rate_mbps": rate,  # 步骤1读取
    "bundle_size": bundle_size,      # 步骤3应用
    "ltp_block_size": ltp_block_size,# 步骤3应用
    "ltp_segment_size": ltp_segment_size,  # 步骤3应用
    "session_count": session_count,   # 步骤3应用
    "model_version": model_version    # 步骤2请求
}
```

---

### 第10步：接收节点B计算业务交付时间

**执行位置**: `receiver.py` → `record_transmission()` 方法

**流程**:
```python
# 步骤10.1：获取接收完成时间（步骤7记录的t2）
t2 = self.transmission_complete_time

# 步骤10.2：从步骤9接收的信息中获取发送开始时间
t1 = transmission_info["transmission_start_time"]

# 步骤10.3：计算业务交付时间
delivery_time_ms = (t2 - t1) * 1000  # 毫秒

print(f"业务交付时间: {delivery_time_ms:.2f}ms")
```

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `record_transmission()` 方法

**关键公式**:
```
delivery_time = t2 - t1
where:
  t1 = 发送开始时间（Sender记录）
  t2 = 接收完成时间（Receiver记录）
```

---

### 第11步：接收节点B的记录器模块生成训练记录

**执行位置**: `receiver.py` → `RecordLogger.add_record()` 方法

**流程**:
```python
# 步骤11.1：收集完整的记录数据

record = {
    # 输入（Input）
    "input": {
        "data_size": 10240,
        "bit_error_rate": 1e-5,
        "delay_ms": 100.0,
        "transmission_rate_mbps": 10.0
    },

    # 输出（Output）
    "output": {
        "bundle_size": 2048,
        "ltp_block_size": 1024,
        "ltp_segment_size": 512,
        "session_count": 3
    },

    # 性能表现（Performance）
    "performance": {
        "delivery_time_ms": 1234.56,  # 业务交付时间
        "throughput_mbps": 13.28,      # 吞吐量 = data_size * 8 / delivery_time / 1M
        "timestamp": 1234567890
    }
}

# 步骤11.2：添加到记录器缓冲区
self.record_logger.add_record(record)

# 步骤11.3：检查是否需要刷新到存储
if self.record_logger.should_flush():
    records = self.record_logger.get_records_to_send()
    # 发送给电脑C（步骤12）
```

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `RecordLogger` 类
- [receiver.py](/root/agent/receive/receiver.py) - `record_transmission()` 方法

**记录格式**:
```json
{
  "input": {
    "data_size": 10240,
    "bit_error_rate": 1e-5,
    "delay_ms": 100.0,
    "transmission_rate_mbps": 10.0
  },
  "output": {
    "bundle_size": 2048,
    "ltp_block_size": 1024,
    "ltp_segment_size": 512,
    "session_count": 3
  },
  "performance": {
    "delivery_time_ms": 1234.56,
    "throughput_mbps": 13.28
  }
}
```

---

### 第12步：接收节点B定期向电脑C发送训练记录

**执行位置**: `receiver.py` → `send_records_to_computer()` 方法

**流程**:
```
Receiver (B)                        Computer C
    │                                    │
    ├─────── TCP 5003 ────────────────►│
    │   JSON批处理:                     │
    │   {                               │
    │     "records": [                 │
    │       {                           │
    │         "input": {...},          │
    │         "output": {...},         │
    │         "performance": {...}     │
    │       },                          │
    │       ...（50条记录）            │
    │     ]                             │
    │   }                               │
    │                                   │◄─接收训练数据
    │◄─────── TCP 5003 ───────────────┤│
    │   确认消息:                        │
    │   {"status": "received"}          │
    │                                   │
```

**触发条件**:
- 缓冲区达到指定数量（默认50条）或
- 经过固定时间间隔（默认5分钟）

**代码位置**:
- [receiver.py](/root/agent/receive/receiver.py) - `send_records_to_computer()` 方法
- [mode_dqn_v2.py](/root/agent/computer/mode_dqn_v2.py) - 训练记录接收服务器

---

### 第13步：电脑C进行DQN训练并更新模型

**执行位置**: `mode_dqn_v2.py` → 训练循环

**流程**:
```python
# 步骤13.1：收到训练记录
received_records = receive_training_records()  # 从TCP 5003端口

# 步骤13.2：对每条记录进行DQN训练
for record in received_records:
    # 提取信息
    input_state = {
        "data_size": record["input"]["data_size"],
        "bit_error_rate": record["input"]["bit_error_rate"],
        "delay_ms": record["input"]["delay_ms"],
        "transmission_rate_mbps": record["input"]["transmission_rate_mbps"]
    }

    output_params = record["output"]  # 当时选择的动作参数
    performance = record["performance"]  # 性能结果

    # 计算奖励
    reward = calculate_reward(
        delivery_time_ms=performance["delivery_time_ms"],
        data_size=input_state["data_size"],
        bit_error_rate=input_state["bit_error_rate"],
        delay_ms=input_state["delay_ms"]
    )

    # 进行DQN更新
    self.store_experience(input_state, action, reward, next_state, done)
    self.train_dqn(batch_size=32)

    # 更新目标网络
    self.network.update_target_network(tau=0.001)

# 步骤13.3：更新模型版本
self.model_version += 1

# 步骤13.4：模型已更新，下一个请求将使用新模型
# 循环回到步骤1...
```

**奖励函数**:
```python
总奖励 = 0.5 × 时间奖励 + 0.3 × 吞吐量奖励 + 0.2 × 鲁棒性奖励

时间奖励:
  if delivery_time ≤ 100ms:   reward = +1.0
  if delivery_time ≥ 5000ms:  reward = -1.0
  else:                        线性插值

吞吐量奖励:
  throughput_mbps = (data_size × 8) / (delivery_time / 1000) / 1M
  reward = 2 × (throughput / 100Mbps) - 1

鲁棒性奖励:
  if BER > 0.5 or delay > 250ms:
    if delivery < 3500ms: reward = +0.5
  else:
    if delivery < 200ms: reward = +0.3
```

**代码位置**:
- [mode_dqn_v2.py](/root/agent/computer/mode_dqn_v2.py) - `train()` 方法
- [mode_dqn_v2.py](/root/agent/computer/mode_dqn_v2.py) - `RewardCalculator` 类

---

### 循环回到步骤1

```
     ┌──────────────────────────────────┐
     │      新的业务请求产生             │
     │                                  │
     │    回到步骤1：发送节点读取CSV     │
     │                                  │
     │    模型版本已更新，参数更优化     │
     │                                  │
     └──────────────┬───────────────────┘
                    │
                    │ 下一轮循环
                    ▼
              ┌─────────────┐
              │  步骤1-13   │
              │ 完整工作流  │
              └─────────────┘
```

---

## 📊 数据流总结

### 消息流向图

```
CSV配置
  │
  ▼
[步骤1] 读取配置和链路状态
  │
  ├─ data_size, BER, delay, rate
  │
  ▼
[步骤2] 向电脑C请求参数
  │
  ├─ 发送: {data_size, link_state}
  ├─ 接收: {bundle, block, segment, session}
  │
  ▼
[步骤3] 应用参数到BP/LTP
  │
  ├─ 更新协议栈参数
  ├─ 配置链路参数
  │
  ▼
[步骤4] 发送元数据给接收端
  │
  ├─ 数据大小、bundle大小等
  │
  ▼
[步骤5] 接收端准备接收
  │
  ├─ 计算bundle数量
  ├─ 开启BP/LTP接收
  │
  ▼
[步骤6-7] 发送和接收数据
  │
  ├─ 记录时间: t1（发送开始）
  ├─ 记录时间: t2（接收完成）
  │
  ▼
[步骤8-9] 传输完成通知和信息回传
  │
  ├─ B通知A完成
  ├─ A发送完整传输信息
  │
  ▼
[步骤10-11] 计算交付时间和生成记录
  │
  ├─ delivery_time = t2 - t1
  ├─ 生成包含输入、输出、性能的记录
  │
  ▼
[步骤12] 定期发送训练记录给电脑C
  │
  ├─ 批量发送50条记录
  ├─ 计时器：5分钟
  │
  ▼
[步骤13] 电脑C进行DQN训练
  │
  ├─ 计算奖励
  ├─ 更新DQN模型
  ├─ 模型版本 += 1
  │
  └─► 回到步骤1（循环）
```

### 时间轴示例

```
时间(毫秒)     Sender A              Receiver B           Computer C
   0           │                     │                    │
   5           ├─ 读CSV配置          │                    │
  10           ├─ 请求参数 ────────────────────────────►│
  15           │                     │                    ├─ 计算参数
  20           │◄────────────────────────────────────────┤
  25           ├─ 应用参数          │                    │
  30           ├─ 发送元数据 ──────────────────────────►│
  35           │                     ├─ 准备接收         │
  40           ├─ 记录t1            │                    │
  45           ├─ 开始发送 ──────────────────────────►│
 100           │                     │ 接收中...          │
1100           │                     ├─ 记录t2           │
1105           │◄─────────────────────┤ 完成通知          │
1110           ├─ 发送传输信息 ──────────────────────►│
1115           │                     ├─ 计算delivery_time │
1120           │                     ├─ 生成记录         │
1130           │                     ├─ (缓冲50条后)      │
1135           │                     ├─ 发送训练记录 ───►│
1140           │                     │                    ├─ 开始训练
1200           │                     │                    ├─ 模型更新
1205           │                     │                    ├─ 版本++
1210           │                     │                    │
 ...           ├─ 新业务请求开始    │                    │
              └─ 循环...
```

---

## 🔧 当前系统实现状态

### ✅ 已实现的功能

| 步骤 | 功能 | 文件 | 状态 |
|------|------|------|------|
| 1 | 读取CSV和链路状态 | sender.py | ✅ 完整 |
| 2 | 参数请求/响应 | sender.py + mode_dqn_v2.py | ✅ 完整 |
| 3 | 应用协议参数 | sender.py + bp_ltp_interface.py | ✅ 完整 |
| 4 | 发送元数据给B | sender.py + receiver.py | ✅ 完整 |
| 5 | 接收端准备 | receiver.py | ✅ 完整 |
| 6-7 | 发送接收数据 | bp_ltp_interface.py | ✅ 完整 |
| 8-9 | 传输完成通知 | sender.py + receiver.py | ✅ 完整 |
| 10-11 | 记录生成 | receiver.py | ✅ 完整 |
| 12 | 训练记录上传 | receiver.py + mode_dqn_v2.py | ✅ 完整 |
| 13 | DQN训练 | mode_dqn_v2.py | ✅ 完整 |

### 📋 系统参数

#### Sender配置（sender.py）
```python
# CSV文件位置: network_config.csv
# CSV字段: sequence, data_size_bytes, bit_error_rate, delay_ms, transmission_rate_mbps, description

# 电脑C地址和端口
optimizer_host = "192.168.1.3"
optimizer_port = 5002  # 参数请求/响应

# 接收节点地址和端口
receiver_host = "192.168.1.2"
receiver_port = 5001   # 元数据和完成通知
```

#### Receiver配置（receiver.py）
```python
# 监听端口
listen_port = 5001     # 接收Sender的元数据和信息
record_port = 5003     # 发送训练记录给Computer C

# 记录器配置
buffer_size = 100      # 缓冲50条记录时上传
flush_interval = 300   # 或每300秒(5分钟)上传
```

#### Computer C配置（mode_dqn_v2.py）
```python
# 监听端口
param_request_port = 5002   # 接收Sender参数请求
record_receive_port = 5003  # 接收Receiver训练记录

# DQN超参数
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1 → 0.01
batch_size = 32
tau = 0.001  # 目标网络软更新

# 动作空间
action_dim = 1064  # (bundle, block, segment)三元组
state_dim = 4      # [data_size, BER, delay, rate]
```

---

## 🚀 启动系统

### 方式1：独立启动（用于调试）

```bash
# 终端1：接收端
cd /root/agent/receive
python3 receiver.py --simulate

# 终端2：优化器
cd /root/agent/computer
python3 mode_dqn_v2.py

# 终端3：发送端
cd /root/agent/send
python3 sender.py --simulate --interval 10
```

### 方式2：集成启动脚本

```bash
#!/bin/bash
# 保存为 start_system.sh

# 启动接收端
python3 /root/agent/receive/receiver.py --simulate > /tmp/receiver.log 2>&1 &
RECEIVER_PID=$!

# 启动优化器
python3 /root/agent/computer/mode_dqn_v2.py > /tmp/optimizer.log 2>&1 &
OPTIMIZER_PID=$!

# 启动发送端
python3 /root/agent/send/sender.py --simulate --interval 5 > /tmp/sender.log 2>&1 &
SENDER_PID=$!

# 保存PID
echo "$RECEIVER_PID $OPTIMIZER_PID $SENDER_PID" > /tmp/pids.txt

echo "系统已启动"
echo "Receiver PID: $RECEIVER_PID"
echo "Optimizer PID: $OPTIMIZER_PID"
echo "Sender PID: $SENDER_PID"

# 监控日志
tail -f /tmp/optimizer.log
```

---

## 📈 监控和调试

### 关键日志消息

| 步骤 | 日志模式 | 文件 |
|------|---------|------|
| 1 | `[CSV配置 N]` | sender.log |
| 2 | `[DQN优化]` | optimizer.log |
| 3 | `[应用参数]` | sender.log |
| 4 | `[发送元数据]` | sender.log |
| 5 | `[准备接收]` | receiver.log |
| 10 | `[记录器] 添加新记录` | receiver.log |
| 12 | `[发送训练记录]` | receiver.log |
| 13 | `[DQN训练] 完成` | optimizer.log |

### 监控命令

```bash
# 查看所有关键事件
grep -E "DQN优化|记录器|发送训练" /tmp/*.log

# 查看奖励变化
grep "平均奖励" /tmp/optimizer.log

# 查看模型版本演进
grep "模型版本" /tmp/optimizer.log | tail -5

# 查看参数选择分布
grep "DQN优化" /tmp/optimizer.log | grep -oP "Bundle=\K\d+" | sort | uniq -c
```

---

## ✅ 工作流程验证清单

- [ ] Sender能读取CSV配置
- [ ] Sender能连接Computer C并请求参数
- [ ] Computer C能返回优化参数
- [ ] Sender能连接Receiver发送元数据
- [ ] Receiver能准备接收
- [ ] 数据能成功传输（BP/LTP）
- [ ] Receiver能记录完成时间
- [ ] Receiver能生成训练记录
- [ ] Receiver能周期性上传记录给Computer C
- [ ] Computer C能接收训练记录
- [ ] Computer C能进行DQN训练
- [ ] 模型版本能递增
- [ ] 新请求使用更新的模型

---

## 📝 总结

本系统通过13个步骤实现了完整的自适应优化循环：

1. **数据采集** (步骤1-4): 收集网络和业务信息
2. **传输执行** (步骤5-9): 执行BP/LTP数据传输
3. **性能评估** (步骤10-11): 计算交付时间和生成记录
4. **模型训练** (步骤12-13): 收集训练数据并更新DQN模型
5. **循环优化** (回到步骤1): 使用更优的参数进行下一轮传输

这个过程不断循环，使系统能够根据实际网络条件和传输性能自动优化BP/LTP协议参数。

**最终目标**：通过强化学习自动调整bundle大小、block大小、segment大小和会话数量，以达到最优的数据传输性能。
