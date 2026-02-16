# 数据流验证和系统测试指南

## 系统数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                      BP/LTP自适应优化系统                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐     ┌──────────────┐│
│  │ 发送节点A    │         │ 优化器C      │     │ 接收节点B    ││
│  │ (Linux板卡)  │         │ (Windows PC) │     │ (Linux板卡)  ││
│  └──────┬───────┘         └──────┬───────┘     └──────┬───────┘│
│         │                        │                     │        │
│  [1] 读取CSV配置                 │                     │        │
│         │                        │                     │        │
│         ├─→ 生成业务请求         │                     │        │
│         │   (data_size)          │                     │        │
│         │                        │                     │        │
│         ├─→ 获取链路状态         │                     │        │
│         │   (BER/延时/速率)      │                     │        │
│         │                        │                     │        │
│  [2] 请求优化参数 ─────────────→│                     │        │
│         │  (data_size+link_state)│                     │        │
│         │                        │                     │        │
│         │  [3]优化参数← ─────────┤                     │        │
│         │   (bundle/block/seg)   │                     │        │
│         │                        │                     │        │
│         ├─→ 应用协议参数         │                     │        │
│         │   (配置LTP)            │                     │        │
│         │                        │                     │        │
│  [4] 发送数据 ──────────────────────────────────→ 接收数据      │
│         │  (start_timestamp)     │                     │        │
│         │  [BP/LTP传输]          │                 [接收]       │
│         │                        │                     │        │
│         ├─→ 发送元数据 ────────────────────────────→[处理]      │
│         │   (data_size+params)   │              计算交付时间    │
│         │                        │                     │        │
│         │                        │         [5] 生成训练记录    │
│         │                        │                     │        │
│  [6] 批量发送训练记录 ─────────────────────────────→│        │
│         │  (input/output/perf)   │                     │        │
│         │                        │                     │        │
│         │         ← ─────── [7]DQN模型更新 ─────────          │
│         │           (版本号+平均奖励)                  │        │
│         │                        │                     │        │
│  [循环周期]                      │                     │        │
│         │                        │                     │        │
│         └────────────────────────┴─────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 接口验证清单

### 1. Sender → Optimizer 参数请求接口

#### 发送端构造的请求
```python
# sender.py:181-241
request_data = {
    "data_size": 20480,                 # ✓ 整数
    "link_state": {                     # ✓ 字典
        "bit_error_rate": 1e-4,         # ✓ 浮点数
        "delay_ms": 150.0,              # ✓ 浮点数
        "transmission_rate_mbps": 8.0   # ✓ 浮点数
    },
    "current_params": {...},            # ✓ 当前参数
    "timestamp": 1707567123.456         # ✓ 时间戳
}
```

#### 优化器处理的请求
```python
# mode_dqn.py:770-800
def handle_param_request(request_data):
    data_size = request_data.get("data_size")
    link_state = request_data.get("link_state", {})

    # 提取状态
    state = {
        "data_size": data_size,
        "bit_error_rate": link_state.get("bit_error_rate"),
        "delay_ms": link_state.get("delay_ms"),
        "transmission_rate_mbps": link_state.get("transmission_rate_mbps")
    }

    # ✓ 完全匹配，无遗漏
    return dqn_optimizer.optimize_params(state)
```

**验证结果**：✅ 100% 匹配

---

### 2. Sender → Receiver 元数据传输接口

#### 发送端构造的元数据
```python
# sender.py:370-403
metadata = {
    "type": "metadata",                 # ✓ 类型标记
    "data_size": 20480,                 # ✓ 数据大小
    "link_state": {                     # ✓ 链路状态完整
        "bit_error_rate": 1e-4,
        "delay_ms": 150.0,
        "transmission_rate_mbps": 8.0
    },
    "protocol_params": {                # ✓ 协议参数完整
        "bundle_size": 2048,
        "ltp_block_size": 1024,
        "ltp_segment_size": 512,
        "session_count": 4
    },
    "timestamp": 1707567123.456         # ✓ 时间戳
}
```

#### 接收端处理的元数据
```python
# receiver.py:278-347
def handle_metadata(data):
    data_size = metadata.get("data_size")                      # ✓ 获得
    link_state = metadata.get("link_state", {})               # ✓ 获得
    protocol_params = metadata.get("protocol_params", {})     # ✓ 获得

    # 所有字段都被提取和使用
    delivery_time_ms = (end_timestamp - start_timestamp) * 1000

    # 生成完整记录
    self.logger.record_transmission(
        data_size=data_size,
        bit_error_rate=link_state.get("bit_error_rate"),
        delay_ms=link_state.get("delay_ms"),
        transmission_rate_mbps=link_state.get("transmission_rate_mbps"),
        bundle_size=protocol_params.get("bundle_size"),
        ltp_block_size=protocol_params.get("ltp_block_size"),
        ltp_segment_size=protocol_params.get("ltp_segment_size"),
        session_count=protocol_params.get("session_count"),
        delivery_time_ms=delivery_time_ms
    )
    # ✓ 所有参数都被传入，无遗漏
```

**验证结果**：✅ 100% 匹配

---

### 3. Receiver → Optimizer 训练数据接口

#### 接收端生成的训练记录
```python
# receiver.py:109-128
record = {
    "input": {                              # ✓ 输入状态
        "data_size": 20480,                 # ✓ 整数
        "bit_error_rate": 1e-4,             # ✓ 浮点数
        "delay_ms": 150.0,                  # ✓ 浮点数
        "transmission_rate_mbps": 8.0       # ✓ 浮点数
    },
    "output": {                             # ✓ 输出动作
        "bundle_size": 2048,                # ✓ 整数
        "ltp_block_size": 1024,             # ✓ 整数
        "ltp_segment_size": 512,            # ✓ 整数
        "session_count": 4                  # ✓ 整数
    },
    "performance": {                        # ✓ 性能指标
        "delivery_time_ms": 1234.56         # ✓ 浮点数
    },
    "timestamp": 1707567124.890             # ✓ 时间戳
}
```

#### 优化器处理的训练记录
```python
# mode_dqn.py:950-1010
def batch_update_model(records):
    for record in records:
        # 解包记录
        input_data = record.get("input", {})           # ✓ 获得
        output_data = record.get("output", {})         # ✓ 获得
        performance = record.get("performance", {})    # ✓ 获得

        # 构造状态
        state = {
            "data_size": input_data.get("data_size"),           # ✓ 获得
            "bit_error_rate": input_data.get("bit_error_rate"), # ✓ 获得
            "delay_ms": input_data.get("delay_ms"),             # ✓ 获得
            "transmission_rate_mbps": input_data.get("transmission_rate_mbps")  # ✓ 获得
        }

        # 计算奖励
        delivery_time_ms = performance.get("delivery_time_ms")  # ✓ 获得
        reward = reward_calculator.calculate_reward(
            delivery_time_ms=delivery_time_ms,
            data_size=input_data.get("data_size"),
            bit_error_rate=input_data.get("bit_error_rate"),
            delay_ms=input_data.get("delay_ms")
        )

        # 查找动作
        action = _find_action_from_params(output_data)  # ✓ 从output_data查找

        # 存储经验
        store_experience(state, action, reward, state, False)  # ✓ 完整

        # 训练
        loss = train_batch()  # ✓ 进行DQN训练
```

**验证结果**：✅ 100% 匹配

---

## 奖励函数数据流

### 输入数据
```
delivery_time_ms: 1234.56
data_size: 20480
bit_error_rate: 1e-4
delay_ms: 150.0
```

### 处理流程

#### 步骤1：时间奖励
```python
if delivery_time_ms ≤ 100:
    time_reward = +1.0
elif delivery_time_ms ≥ 5000:
    time_reward = -1.0
else:
    time_ratio = (1234.56 - 100) / (5000 - 100) = 0.2559
    time_reward = 1.0 - 2.0 × 0.2559 = 0.488
```
**输出**：time_reward = 0.488

#### 步骤2：吞吐量奖励
```python
throughput_mbps = (20480 × 8) / (1234.56 / 1000) / 1_000_000
               = 163840 / 1234.56 / 1_000_000
               = 0.1327 Mbps  # 很低

throughput_normalized = min(0.1327 / 100.0, 1.0) = 0.001327
throughput_reward = 2.0 × 0.001327 - 1.0 = -0.9973
```
**输出**：throughput_reward = -0.9973

#### 步骤3：鲁棒性奖励
```python
ber_factor = min(1e-4 × 1e6, 1.0) = min(0.1, 1.0) = 0.1
delay_factor = min(150.0 / 500.0, 1.0) = 0.3
adversity = (0.1 + 0.3) / 2.0 = 0.2

adversity < 0.5 → 良好条件
if delivery_time_ms < min_time × 2:  (100 × 2 = 200)
    → 1234.56 < 200? NO
    → return 0.0

robustness_reward = 0.0
```
**输出**：robustness_reward = 0.0

#### 步骤4：加权组合
```python
total_reward = 0.5 × 0.488 + 0.3 × (-0.9973) + 0.2 × 0.0
            = 0.244 - 0.299 + 0
            = -0.055
```

**最终输出**：reward = -0.055

### 含义解释

**负奖励** (-0.055) 表示：
- ❌ 交付时间较长（1234.56ms > 100ms）
- ❌ 吞吐量过低（0.1327 Mbps）
- ➖ 链路条件中等，无特别表现

**系统会倾向于选择能减少交付时间的参数**。

---

## 完整传输周期数据流追踪

### 周期 #1

#### T=0.0s: Sender读取CSV
```
CSV行 #3: data_size=5120, BER=1e-3, delay=200ms, rate=5Mbps
```

#### T=0.1s: Sender请求参数
```
→ Optimizer: {
    "data_size": 5120,
    "link_state": {"bit_error_rate": 1e-3, "delay_ms": 200.0, "transmission_rate_mbps": 5.0},
    ...
}
```

#### T=0.15s: Optimizer生成参数
```
DQN处理：
  状态向量: [5120/100000, 1e-3×1e6, 200/500, 5/100] = [0.0512, 1.0, 0.4, 0.05]
  动作选择: ε=0.1 → 10% 探索 / 90% 利用 → action=7
  参数: {bundle_size: 1024, ltp_block: 512, ltp_seg: 256, session: 2}

← Sender: {
    "optimized_params": {...},
    "model_version": 5,
    ...
}
```

#### T=0.2s: Sender发送元数据给Receiver
```
→ Receiver: {
    "type": "metadata",
    "data_size": 5120,
    "link_state": {...},
    "protocol_params": {...},
    ...
}
```

#### T=0.3s: Receiver处理元数据
```
start_timestamp: 0.2 (发送开始时间)
end_timestamp: 0.3 (接收完成时间)
delivery_time_ms: (0.3 - 0.2) × 1000 = 100ms

生成记录: {
    "input": {"data_size": 5120, "bit_error_rate": 1e-3, ...},
    "output": {"bundle_size": 1024, ...},
    "performance": {"delivery_time_ms": 100},
    ...
}
```

#### T=60.0s: Receiver发送50条记录给Optimizer
```
→ Optimizer: {
    "type": "training_records",
    "records": [record1, record2, ..., record50],
    "count": 50,
    ...
}
```

#### T=60.5s: Optimizer批量更新DQN模型
```
对每条record:
  计算奖励: reward = 0.5×time + 0.3×throughput + 0.2×robustness
  存储经验: (state, action, reward, next_state, done)
  训练网络: 若buffer满，采样批次进行反向传播

更新完成:
  模型版本: 6 (从5→6)
  平均奖励: 0.324
  探索率ε: 0.0995 (0.1 × 0.995)
```

### 周期 #2

#### T=60.0s: Sender循环，读取CSV行 #4
```
CSV行 #4: data_size=40960, BER=5e-5, delay=80ms, rate=15Mbps
```

#### T=60.1s: Sender再次请求参数
```
→ Optimizer: {
    "data_size": 40960,
    "link_state": {"bit_error_rate": 5e-5, "delay_ms": 80.0, "transmission_rate_mbps": 15.0},
    ...
}
```

#### T=60.15s: Optimizer生成新参数（使用更新的DQN模型）
```
DQN处理：
  状态向量: [40960/100000, 5e-5×1e6, 80/500, 15/100] = [0.41, 0.05, 0.16, 0.15]
  动作选择: ε=0.0995 → 9.95% 探索 / 90.05% 利用 → action=12 (不同了！)
  参数: {bundle_size: 4096, ltp_block: 512, ltp_seg: 256, session: 8}

← Sender: {
    "optimized_params": {...},  # 不同的参数组合！
    "model_version": 6,  # 版本已更新
    ...
}
```

...继续循环

---

## 测试验证步骤

### 测试1：启动系统并验证连接

```bash
# 终端1: 启动接收端
python3 /root/agent/receive/receiver.py --simulate

# 终端2: 启动优化器（DQN）
python3 /root/agent/computer/mode_dqn.py

# 终端3: 启动发送端
python3 /root/agent/send/sender.py --simulate
```

**验证项**：
- [ ] 三个进程正常启动，无错误
- [ ] 接收端监听 5001 端口
- [ ] 优化器监听 5002 和 5003 端口
- [ ] 发送端连接建立

### 测试2：验证参数流（Sender → Optimizer）

**预期日志**：
```
[发送端] 已发送请求到优化器 192.168.1.3:5002
[优化器] [新请求] 来自 192.168.1.1, 模型版本: 0
[优化器] [参数请求] 数据量: 10240 bytes, 误码率: 1e-05, ...
[优化器] [参数响应] 已发送优化参数: {'bundle_size': 1024, ...}
[发送端] [收到优化参数] {'bundle_size': 1024, ...}
```

**验证命令**：
```bash
# 检查网络连接
netstat -tlnp | grep -E "5002"

# 检查日志
grep "参数请求\|参数响应" optimizer.log
grep "收到优化参数" sender.log
```

### 测试3：验证元数据流（Sender → Receiver）

**预期日志**：
```
[发送端] [元数据发送] 已发送传输元数据到节点B
[接收端] [新连接] 来自 192.168.1.1
[接收端] [元数据接收] 完成时间戳: 2026-02-10 12:34:57.123456
[接收端] [传输指标] 业务交付时间: 1234.56ms
[记录器] 添加新记录 (当前缓冲数: 1/100)
```

**验证命令**：
```bash
# 检查元数据接收
grep "元数据接收\|传输指标" receiver.log

# 检查记录生成
grep "记录器.*添加新记录" receiver.log
```

### 测试4：验证训练数据流（Receiver → Optimizer）

**预期日志**：
```
[接收端] [记录发送] 成功发送 50 条记录到优化器
[优化器] [收到训练记录] 50 条来自 192.168.1.2
[DQN训练] 开始使用 50 条记录进行批量训练
[DQN训练] [记录1/50] 奖励: 0.4523, 交付时间: 1234.56ms, Loss: 0.003421
[DQN训练完成] 模型版本: 1, 平均奖励: 0.3891, 探索率ε: 0.0995
```

**验证命令**：
```bash
# 检查记录发送
grep "记录发送" receiver.log

# 检查DQN训练
grep "DQN训练\|模型版本" optimizer.log

# 查看平均奖励变化
grep "平均奖励" optimizer.log | tail -5
```

### 测试5：验证DQN模型改进

**验证步骤**：
```bash
# 运行系统1小时，收集数据
timeout 3600 python3 /root/agent/send/sender.py --simulate --interval 10

# 检查模型版本号演进
grep "模型版本" optimizer.log | awk -F': ' '{print $2}' | sort -u

# 检查平均奖励演进
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | paste -sd ',' - | tr ',' '\n' | sort -n | tail -5
```

**预期结果**：
- 模型版本号递增（0 → 1 → 2 → ... → N）
- 平均奖励逐步提升（-0.2 → -0.1 → 0.1 → 0.3 → 0.4）
- 探索率逐步下降（0.1 → 0.099 → ... → 0.01）

### 测试6：参数多样性检查

**验证优化器是否学会不同条件下使用不同参数**：

```python
# 分析日志，提取所有返回的参数组合
import re

with open('optimizer.log', 'r') as f:
    lines = f.readlines()

params_used = []
for line in lines:
    if '已发送优化参数' in line:
        # 提取参数
        match = re.search(r"'bundle_size': (\d+).*'session_count': (\d+)", line)
        if match:
            params_used.append((match.group(1), match.group(2)))

# 统计不同参数组合的使用频率
from collections import Counter
param_freq = Counter(params_used)
print("参数组合使用频率:")
for param, freq in param_freq.most_common():
    print(f"  Bundle={param[0]}, Session={param[1]}: {freq} 次")
```

**预期结果**：
- 在不同网络条件下，使用不同的参数组合
- 不是单一重复使用同一个参数，而是动态调整
- 在恶劣条件下倾向于用较小的bundle
- 在良好条件下倾向于用较大的bundle

---

## 故障排除

### 问题1：Sender无法连接Optimizer

```
[错误] 请求优化参数失败: [Errno 111] Connection refused
```

**检查**：
```bash
# 1. 确认优化器是否运行
ps aux | grep mode_dqn

# 2. 检查端口是否开放
netstat -tlnp | grep 5002

# 3. 检查防火墙
iptables -L | grep 5002
```

### 问题2：Receiver未生成训练记录

```
[警告] 没有要发送的记录
```

**检查**：
```bash
# 1. 确认元数据是否被接收
grep "元数据接收" receiver.log | wc -l

# 2. 检查记录生成
grep "记录器.*添加新记录" receiver.log | wc -l

# 3. 验证缓冲区状态
grep "缓冲数" receiver.log | tail -1
```

### 问题3：DQN奖励始终为负

**分析**：
```bash
# 查看交付时间分布
grep "交付时间" optimizer.log | awk -F': ' '{print $3}' | tr 'ms' ' ' | awk '{print $1}' | sort -n

# 检查吞吐量
grep "吞吐量奖励" optimizer.log  # 如果添加了详细日志
```

**可能原因**：
- 交付时间过长（> 5000ms）
- 网络条件差
- bundle大小配置不当

**解决方案**：
- 增加发送端等待时间
- 调整奖励权重，减少对交付时间的要求
- 使用不同的network_config.csv

---

## 性能指标汇总

| 指标 | 预期值 | 实际值 | 状态 |
|------|--------|--------|------|
| Sender → Optimizer 延时 | < 100ms |  |  |
| 平均交付时间 | < 2000ms |  |  |
| 模型收敛时间 | < 1小时 |  |  |
| 最终平均奖励 | > 0.2 |  |  |
| 参数多样性 | > 5种组合 |  |  |

---

本文档提供了完整的数据流验证和测试方案。按照步骤执行，可以确保系统各部分正常协作！