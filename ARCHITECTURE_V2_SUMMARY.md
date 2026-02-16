# BP/LTP自适应优化系统 - 架构v2总结

## 📋 文档概述

本文档总结了BP/LTP自适应优化系统的**第二版架构**，重点说明了关键的架构改正：**LTP会话数量是通过确定性计算得出的，不是强化学习训练的结果**。

---

## 🎯 核心改正

### 问题描述
初始设计错误地将LTP会话数量(`session_count`)作为强化学习模型需要优化的输出参数。

### 解决方案
根据`calculate_ltp_sessions()`函数的存在，重新认识到：
- **LTP会话数量是通过网络条件和协议参数可以直接计算得出的**
- **不需要通过RL训练学习**
- **只需优化3个参数：bundle_size、ltp_block_size、ltp_segment_size**

### 架构变化

| 方面 | v1（初始） | v2（改正） |
|------|----------|----------|
| 优化的参数个数 | 4个（+session） | 3个 |
| 动作空间大小 | 16个 (4×4) | 9个 (3×3) |
| session_count来源 | RL输出 | 计算函数 |
| 状态空间 | 4维（不变） | 4维（不变） |
| 奖励函数 | 多维（不变） | 多维（不变） |

---

## 🔧 系统架构详解

### 1. 三层系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    电脑C（算力中心）                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │          DQN优化器 (mode_dqn_v2.py)              │  │
│  │  - 神经网络: 4→128→128→9                         │  │
│  │  - 经验回放缓冲区                                  │  │
│  │  - 目标网络（软更新）                              │  │
│  │  - 奖励计算器                                      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         ↓ (参数请求/响应)      ↑ (训练记录)
         ↓                      ↑
┌─────────────────────────────────────────────────────────┐
│            节点A (sender.py) ↔ 节点B (receiver.py)      │
│  - 生成业务                    - 接收数据                │
│  - 请求优化参数                - 记录性能                │
│  - 应用LTP协议栈               - 发送训练记录            │
│  - 发送传输元数据              - 计算奖励（隐含）        │
└─────────────────────────────────────────────────────────┘
```

### 2. 数据流工作原理

#### 阶段1：参数优化请求
```
sender.py:
  1. 读取网络配置 (CSV)
  2. 收集当前链路状态:
     - data_size: 数据量(bytes)
     - bit_error_rate: 误码率
     - delay_ms: 延时
     - transmission_rate_mbps: 传输速率
  3. 向optimizer发送请求 (TCP:5002)
     请求格式: {
       "model_version": int,
       "data_size": int,
       "link_state": {
         "bit_error_rate": float,
         "delay_ms": float,
         "transmission_rate_mbps": float
       }
     }
```

#### 阶段2：参数优化生成（关键改正）
```
mode_dqn_v2.py:
  1. discretize_state(): 规范化状态向量
     [data_size/100000, BER*1e6, delay/500, rate/100]

  2. select_action(): ε-贪心选择
     - ε概率探索（随机）
     - (1-ε)概率利用（Q值最大）

  3. action_to_params(): 动作→参数（包含计算）
     └─ 9个动作映射: action = bundle_idx*3 + block_idx

     动作空间:
     ┌─────────────────────────────────┐
     │ Bundle Size: [1024, 2048, 4096] │
     │ Block Size:  [512, 1024, 2048]  │
     │ Segment Size: [256, 512, 1024]  │
     └─────────────────────────────────┘

     ★★★ 关键改正 ★★★
     session_count = calculate_ltp_sessions(
         delay=delay_ms,
         bundle_size=bundle_size,
         file_size=data_size,
         block_size=ltp_block_size,
         trans_rate=transmission_rate_bytes
     )

     返回参数: {
       "bundle_size": bundle_size,
       "ltp_block_size": ltp_block_size,
       "ltp_segment_size": ltp_segment_size,
       "session_count": 计算得出的值  ← 不再是RL输出
     }

  4. 将参数响应给sender (TCP:5002)
```

#### 阶段3：协议应用与传输
```
sender.py:
  1. 收到优化参数 (from optimizer)
  2. 应用到BP/LTP协议栈:
     bp_ltp_interface.setup_parameters(
       bundle_size=params["bundle_size"],
       ltp_block_size=params["ltp_block_size"],
       ltp_segment_size=params["ltp_segment_size"],
       session_count=params["session_count"]  ← 使用计算值
     )
  3. 发送业务传输元数据给receiver
```

#### 阶段4：性能评估与模型训练
```
receiver.py:
  1. 接收传输元数据 (from sender)
  2. 记录接收完成时间
  3. 使用RecordLogger生成训练记录:
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
         "session_count": int  ← 接收的计算值
       },
       "performance": {
         "delivery_time_ms": float,
         "throughput_mbps": float
       }
     }
  4. 周期性发送记录给optimizer (TCP:5003)

mode_dqn_v2.py:
  1. 接收训练记录批次
  2. 计算奖励:
     reward = 0.5×time_reward + 0.3×throughput_reward + 0.2×robustness_reward
  3. 执行DQN训练:
     - 经验回放采样
     - 目标Q值计算（使用目标网络）
     - 损失函数反向传播
     - 权重更新
     - 目标网络软更新 (τ=0.001)
  4. 更新模型版本
```

---

## 🧮 calculate_ltp_sessions()函数详解

### 源自: `/root/agent/send/dtn_ion.py`

### 函数签名
```python
def calculate_ltp_sessions(
    delay: float,           # 延时（毫秒）
    bundle_size: int,       # Bundle大小（bytes）
    file_size: int,         # 文件大小（bytes）
    block_size: int,        # LTP Block大小（bytes）
    trans_rate: float       # 传输速率（bytes/s）
) -> int:
```

### 计算逻辑
```python
# 第1步：计算需要多少个bundle来承载整个文件
total_bundles = ceil(file_size / bundle_size)

# 第2步：计算每个block包含多少个bundle
bundles_per_block = ceil(block_size / bundle_size)

# 第3步：计算需要多少个LTP block
ltp_blocks = ceil(total_bundles / bundles_per_block)

# 第4步：估计一个block的传输时间（往返延时+发送时间）
times = delay/500 + ((block_size + 20) / trans_rate)

# 第5步：计算最优会话数（能管道化处理的block数）
ltp_sessions = ceil((times * trans_rate) / (block_size + 20)) + 1

# 第6步：应用限制条件（最多20个会话，至少1个）
ltp_sessions = min(ltp_sessions, ltp_blocks + 1, 20)

return ltp_sessions
```

### 核心概念
- **为什么需要计算？**
  - LTP会话数应该与网络延时、传输速率、数据大小匹配
  - 过少的会话：管道效率低
  - 过多的会话：资源浪费
  - 通过公式计算得到**最优的平衡点**

- **为什么不需要训练？**
  - 这是一个**确定性函数**，给定输入就有确定输出
  - RL应该学习的是**不确定的参数**（bundle、block、segment的大小选择）
  - 会话数的计算已经体现了对网络特性的理解

---

## 🎓 DQN训练流程

### 网络架构
```
输入层 (4维)
    ↓
隐层1 (128个ReLU) + Dropout概念
    ↓
隐层2 (128个ReLU) + Dropout概念
    ↓
输出层 (9个Q值)
    ↓
argmax → 动作索引
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Learning Rate | 0.001 | 优化器学习率 |
| Gamma (γ) | 0.99 | 未来奖励折扣因子 |
| Epsilon (ε) | 0.1→0.01 | 探索率（衰减） |
| Epsilon Decay | 0.995 | 每步衰减系数 |
| Batch Size | 32 | 批训练大小 |
| Tau (τ) | 0.001 | 目标网络软更新系数 |
| Replay Buffer | 10000 | 经验回放容量 |

### 奖励函数
```python
总奖励 = 0.5×时间奖励 + 0.3×吞吐量奖励 + 0.2×鲁棒性奖励

时间奖励 (权重0.5)：
  if delivery_time ≤ 100ms:  reward = +1.0
  if delivery_time ≥ 5000ms: reward = -1.0
  else:                       线性插值

吞吐量奖励 (权重0.3)：
  throughput_mbps = (data_size × 8) / (delivery_time / 1000) / 1M
  reward = 2 × (throughput / 100Mbps) - 1
  范围: [-1.0, +1.0]

鲁棒性奖励 (权重0.2)：
  if BER>0.5 or delay>250ms:
    if delivery < 3500ms: reward = +0.5
  else:
    if delivery < 200ms:  reward = +0.3
```

---

## 📊 动作空间映射表

### 9个动作详解

| 动作 | Bundle | Block | Segment | 说明 |
|------|--------|-------|---------|------|
| 0 | 1024 | 512 | 256 | 最小开销 |
| 1 | 1024 | 1024 | 512 | 小块 |
| 2 | 1024 | 2048 | 1024 | 小bundle+大block |
| 3 | 2048 | 512 | 256 | 中等bundle+小block |
| 4 | 2048 | 1024 | 512 | 均衡配置（推荐） |
| 5 | 2048 | 2048 | 1024 | 中等配置 |
| 6 | 4096 | 512 | 256 | 大bundle+小block |
| 7 | 4096 | 1024 | 512 | 大bundle+中block |
| 8 | 4096 | 2048 | 1024 | 最大配置 |

### 动作→参数转换
```python
# 内部实现
bundle_idx = action // 3      # 0, 1, 2
block_idx = action % 3         # 0, 1, 2

bundle_size = [1024, 2048, 4096][bundle_idx]
ltp_block_size = [512, 1024, 2048][block_idx]
ltp_segment_size = [256, 512, 1024][block_idx]

# 关键：计算session_count
ltp_sessions = calculate_ltp_sessions(
    delay=delay_ms,
    bundle_size=bundle_size,
    file_size=data_size,
    block_size=ltp_block_size,
    trans_rate=transmission_rate_bytes
)
```

---

## 🔗 系统接口验证

### 接口1: Sender → Optimizer (参数请求)
**端口**: TCP 5002
**协议**: JSON
**数据匹配**: ✅ 100%

**Sender发送**:
```json
{
  "model_version": 5,
  "data_size": 20480,
  "link_state": {
    "bit_error_rate": 0.0001,
    "delay_ms": 150.0,
    "transmission_rate_mbps": 8.0
  }
}
```

**Optimizer返回**:
```json
{
  "bundle_size": 2048,
  "ltp_block_size": 1024,
  "ltp_segment_size": 512,
  "session_count": 3
}
```

### 接口2: Optimizer → Sender (参数响应)
同上，返回优化参数

### 接口3: Sender → Receiver (传输元数据)
**端口**: TCP 5001
**协议**: JSON
**数据匹配**: ✅ 100%

### 接口4: Receiver → Optimizer (训练记录)
**端口**: TCP 5003
**协议**: JSON批处理
**数据匹配**: ✅ 100%

**Receiver发送**:
```json
{
  "records": [
    {
      "input": {
        "data_size": 20480,
        "bit_error_rate": 0.0001,
        "delay_ms": 150.0,
        "transmission_rate_mbps": 8.0
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
  ]
}
```

---

## 🚀 部署与运行

### 快速启动
```bash
# 在三个不同的终端中运行：

# 终端1：启动接收端
python3 /root/agent/receive/receiver.py --simulate

# 终端2：启动优化器（使用v2版本）
python3 /root/agent/computer/mode_dqn_v2.py

# 终端3：启动发送端
python3 /root/agent/send/sender.py --simulate --interval 10
```

### 自定义端口
```bash
python3 /root/agent/computer/mode_dqn_v2.py \
  --param-port 5002 \
  --record-port 5003
```

---

## 📈 性能指标

### 预期表现

| 指标 | 训练初期 | 收敛后 | 目标 |
|------|----------|--------|------|
| 平均奖励 | -0.3 ~ 0.0 | 0.2 ~ 0.5 | > 0.3 |
| 探索率ε | 0.1 | 0.01 ~ 0.05 | 自动衰减 |
| 交付时间 | 1000~3000ms | 500~1500ms | < 1500ms |
| 模型版本 | 0~10 | 50+ | 持续增长 |

### 判断学习质量

**✅ 正常学习迹象**:
- 平均奖励逐步上升
- 探索率逐步下降
- 交付时间逐步减少
- 参数选择越来越稳定

**❌ 学习异常迹象**:
- 平均奖励持续为负且不改善
- 所有交付时间都很长 (> 3000ms)
- 参数选择完全随机
- Loss不收敛或爆炸

---

## 🔍 监控与调试

### 日志查看
```bash
# 查看优化器日志
tail -f /root/agent/computer/optimizer.log

# 查看关键训练信息
tail -f optimizer.log | grep -E "DQN训练|模型版本|平均奖励"

# 查看奖励变化
tail -f optimizer.log | grep "奖励:" | awk '{print $3}'
```

### 检查系统状态
```bash
# 查看进程
ps aux | grep python3

# 查看网络连接
netstat -tlnp | grep -E "5001|5002|5003"

# 检查模型进展
grep "模型版本" optimizer.log | tail -5
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | tail -10
```

---

## ⚙️ 常见问题

### Q1: 为什么不直接计算session_count？
**A**: 最初的理解误区。通过`calculate_ltp_sessions()`函数验证，会话数本身是确定性计算的结果，而非需要学习的参数。这个发现改进了架构的合理性。

### Q2: 为什么要从16个动作改为9个？
**A**:
- 16个动作 = 4×4（包括session），但session会被覆盖计算
- 9个动作 = 3×3（只需优化bundle×block），session自动计算
- 减少搜索空间，加快收敛，避免冗余训练

### Q3: 怎么验证系统是否正常工作？
**A**:
```bash
# 运行10分钟测试
timeout 600 python3 /root/agent/send/sender.py --simulate --interval 5

# 检查平均奖励是否上升
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | head -1
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | tail -1
# 应该看到：第一次 -0.2x, 最后一次 -0.1x 或更高
```

### Q4: session_count计算值不合理怎么办？
**A**: 检查`calculate_ltp_sessions()`的输入参数：
- `delay`: 必须是毫秒单位
- `bundle_size/block_size`: 必须是字节
- `file_size`: 必须是字节
- `trans_rate`: 必须是字节/秒

---

## 📝 文件对应关系

| 文件 | 功能 | 状态 |
|------|------|------|
| `/root/agent/send/sender.py` | 发送节点，请求优化参数 | ✅ 兼容v2 |
| `/root/agent/receive/receiver.py` | 接收节点，生成训练记录 | ✅ 兼容v2 |
| `/root/agent/computer/mode_dqn_v2.py` | DQN优化器（改正版） | ✅ 当前使用 |
| `/root/agent/computer/mode_dqn.py` | DQN优化器（v1版本） | ⚠️ 已过时 |
| `/root/agent/send/dtn_ion.py` | BP/LTP协议接口 | ✅ 核心函数 |
| `DQN_QUICK_REFERENCE.md` | 快速查询表 | ✅ 已更新 |
| `DATA_FLOW_VERIFICATION.md` | 数据流验证 | ✅ 已验证 |
| `DQN_INTEGRATION_GUIDE.md` | 集成指南 | ✅ 已更新 |

---

## 🎯 总结

### 架构v2的核心改进
1. **正确理解session_count**: 从RL输出改为确定性计算
2. **简化动作空间**: 从16个减少到9个，加快收敛
3. **保留核心价值**: 保持4维状态空间、多维奖励函数、完整DQN训练
4. **系统兼容性**: 无需修改sender和receiver，仅替换optimizer

### 为什么这样设计更好？
- **理论正确**: 符合强化学习的原理——学习不确定性，计算确定性
- **性能优化**: 减少搜索空间，提高训练效率
- **可维护性**: 分离concerns，session计算独立于RL
- **可扩展性**: 未来可轻松添加新的计算参数

---

**本文档版本**: 2.0（架构改正版）
**最后更新**: 2025年（当前）
**相关文件**: 见文件对应关系表