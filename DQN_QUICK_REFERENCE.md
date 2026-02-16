# DQN优化器快速参考

## 📊 系统接口验证总结

| 接口 | 状态 | 数据匹配度 |
|------|------|------------|
| Sender → Optimizer (参数请求) | ✅ | 100% |
| Optimizer → Sender (参数响应) | ✅ | 100% |
| Sender → Receiver (元数据) | ✅ | 100% |
| Receiver → Optimizer (训练数据) | ✅ | 100% |

## 🎯 奖励函数公式

```
总奖励 = 0.5 × 时间奖励 + 0.3 × 吞吐量奖励 + 0.2 × 鲁棒性奖励
```

### 时间奖励（权重0.5）
```
if delivery_time ≤ 100ms:    reward = +1.0 (最佳)
if delivery_time ≥ 5000ms:   reward = -1.0 (最差)
else:                         线性插值
```

### 吞吐量奖励（权重0.3）
```
throughput_mbps = (data_size × 8) / (delivery_time / 1000) / 1M
reward = 2 × (throughput / 100Mbps) - 1
范围: [-1.0, +1.0]
```

### 鲁棒性奖励（权重0.2）
```
恶劣条件 (BER>0.5 or delay>250ms):
  if delivery < 3500ms:  reward = +0.5
良好条件:
  if delivery < 200ms:   reward = +0.3
```

## 🏗️ DQN架构

```
状态输入 (4维)
    ↓
FC层1 (128神经元) + ReLU
    ↓
FC层2 (128神经元) + ReLU
    ↓
输出层 (16个Q值)
    ↓
argmax → 动作索引
    ↓
映射 → 协议参数
```

## ⚙️ 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 0.001 | 网络学习率 |
| `gamma` | 0.99 | 未来奖励折扣 |
| `epsilon` | 0.1 → 0.01 | 探索率（衰减） |
| `epsilon_decay` | 0.995 | 探索率衰减系数 |
| `batch_size` | 32 | 批训练大小 |
| `tau` | 0.001 | 目标网络软更新 |
| `replay_buffer` | 10000 | 经验回放容量 |

## 📈 状态和动作空间

### 状态空间（4维）
```
s = [data_size/100000, BER×1e6, delay/500, rate/100]
归一化范围: [0, 1] × [0, 1] × [0, 1] × [0, 1]
```

### 动作空间（16个离散动作）
```
action = bundle_idx × 4 + block_idx

其中:
bundle_idx ∈ {0,1,2,3} → {512, 1024, 2048, 4096}
block_idx  ∈ {0,1,2,3} → {256, 512, 1024, 2048}
```

### 动作映射快查表

| 动作 | Bundle | Block | Segment | Session |
|------|--------|-------|---------|---------|
| 0    | 512    | 256   | 128     | 1       |
| 5    | 1024   | 512   | 256     | 2       |
| 10   | 2048   | 1024  | 512     | 4       |
| 15   | 4096   | 2048  | 1024    | 8       |

## 🚀 启动命令

```bash
# 启动DQN优化器
python3 /root/agent/computer/mode_dqn.py

# 自定义端口
python3 /root/agent/computer/mode_dqn.py \
  --param-port 5002 \
  --record-port 5003
```

## 🔍 监控命令

### 实时查看日志
```bash
# 查看优化器日志
tail -f /root/agent/computer/optimizer.log

# 过滤DQN训练信息
tail -f optimizer.log | grep -E "DQN训练|模型版本|平均奖励"

# 查看奖励变化
tail -f optimizer.log | grep "奖励:" | awk '{print $3}'
```

### 检查模型进展
```bash
# 模型版本演进
grep "模型版本" optimizer.log | tail -5

# 平均奖励趋势
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | tail -10

# 探索率衰减
grep "探索率" optimizer.log | awk -F': ' '{print $2}' | tail -10
```

### 分析参数使用情况
```bash
# 统计各参数组合使用频率
grep "已发送优化参数" optimizer.log | \
  grep -oP "'bundle_size': \d+" | \
  sort | uniq -c
```

## 📊 性能指标

### 预期指标值

| 指标 | 训练初期 | 收敛后 | 目标 |
|------|----------|--------|------|
| 平均奖励 | -0.3 ~ 0.0 | 0.2 ~ 0.5 | > 0.3 |
| 探索率ε | 0.1 | 0.01 ~ 0.05 | 自动衰减 |
| 交付时间 | 1000~3000ms | 500~1500ms | < 1500ms |
| 模型版本 | 0~10 | 50+ | 持续增长 |

### 判断模型是否在学习

**✅ 正常学习迹象**：
- 平均奖励逐步上升
- 探索率逐步下降
- 交付时间逐步减少
- 参数选择越来越稳定

**❌ 学习异常迹象**：
- 平均奖励持续为负且不改善
- 所有交付时间都很长 (> 3000ms)
- 参数选择完全随机
- Loss不收敛或爆炸

## 🔧 快速调优

### 如果奖励太低（< -0.5）

**调整奖励权重**：
```python
# 在 mode_dqn.py 的 calculate_reward() 中
total_reward = (
    0.3 * time_reward +      # 降低时间权重
    0.5 * throughput_reward + # 增加吞吐量权重
    0.2 * robustness_reward
)
```

### 如果学习太慢

**增加学习率**：
```python
# 在 DQNNetwork.__init__() 中
self.learning_rate = 0.01  # 从0.001提高到0.01
```

**增加探索率**：
```python
# 在 DQNOptimizer.__init__() 中
self.epsilon = 0.3  # 从0.1提高到0.3
self.epsilon_decay = 0.99  # 从0.995提高到0.99
```

### 如果参数不收敛

**减少探索率**：
```python
self.epsilon = 0.05  # 减少探索
self.epsilon_min = 0.001  # 更低的最小值
```

**增加批次大小**：
```python
self.batch_size = 64  # 从32提高到64
```

## 🧪 快速测试

### 测试1：单周期完整流程（1分钟）
```bash
# 启动所有组件
python3 /root/agent/receive/receiver.py --simulate &
python3 /root/agent/computer/mode_dqn.py &
python3 /root/agent/send/sender.py --simulate --interval 10

# 观察输出60秒
sleep 60

# 检查是否正常
ps aux | grep python3
netstat -tlnp | grep -E "5001|5002|5003"
```

### 测试2：验证数据流（30秒）
```bash
# 等待发送一次
sleep 30

# 检查关键日志
grep "收到优化参数" /tmp/sender.log
grep "元数据接收" /tmp/receiver.log
grep "DQN训练" /tmp/optimizer.log
```

### 测试3：检查模型学习（10分钟）
```bash
# 运行10分钟
timeout 600 python3 /root/agent/send/sender.py --simulate --interval 5

# 检查平均奖励是否上升
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | head -1
grep "平均奖励" optimizer.log | awk -F': ' '{print $2}' | tail -1

# 应该看到:
# 第一次: -0.2x
# 最后一次: -0.1x 或更高
```

## 🐛 常见问题速查

### Q1: ImportError: No module named 'numpy'
```bash
pip3 install numpy
```

### Q2: 奖励始终为 -1.0
**原因**: 交付时间超过5000ms

**解决**:
```python
# 调整 RewardCalculator 中的范围
max_delivery_time_ms = 10000.0  # 改为10秒
```

### Q3: 模型版本不增长
**原因**: 没有收到训练记录

**检查**:
```bash
# 确认接收端正常工作
grep "记录器.*添加" receiver.log

# 确认发送成功
grep "记录发送.*成功" receiver.log
```

### Q4: 网络连接失败
**检查防火墙**:
```bash
# 临时关闭防火墙测试
sudo ufw disable

# 或开放端口
sudo ufw allow 5001:5003/tcp
```

## 📝 日志示例

### 正常运行的日志片段

```
[新请求] 来自 192.168.1.1, 模型版本: 5
[参数请求] 数据量: 20480 bytes, 误码率: 0.0001, 延时: 150.0ms, 速率: 8.0Mbps
[参数响应] 已发送优化参数: {'bundle_size': 2048, 'ltp_block_size': 1024, ...}

[收到训练记录] 50 条来自 192.168.1.2
[DQN训练] 开始使用 50 条记录进行批量训练
  [记录1/50] 奖励: 0.4523, 交付时间: 1234.56ms, Loss: 0.003421
  [记录11/50] 奖励: 0.3891, 交付时间: 1456.78ms, Loss: 0.002987
  ...
[DQN训练完成] 模型版本: 6, 平均奖励: 0.4127, 探索率ε: 0.0995, 训练步数: 192
```

## 🎓 关键概念提醒

1. **DQN = Deep Q-Network**
   - 使用神经网络代替Q表
   - 能处理连续状态空间
   - 适合复杂环境

2. **经验回放**
   - 打破时间相关性
   - 提高样本效率
   - 稳定训练过程

3. **目标网络**
   - 独立于主网络
   - 缓慢更新（τ=0.001）
   - 提供稳定的学习目标

4. **ε-贪心策略**
   - ε概率探索（随机）
   - (1-ε)概率利用（最优）
   - ε逐步衰减

5. **多维奖励**
   - 平衡不同目标
   - 加权组合
   - 归一化范围

---

**需要更多帮助？**

- 详细文档: [DQN_INTEGRATION_GUIDE.md](DQN_INTEGRATION_GUIDE.md)
- 数据流验证: [DATA_FLOW_VERIFICATION.md](DATA_FLOW_VERIFICATION.md)
- 系统概览: [SYSTEM_INTEGRATION_SUMMARY.md](SYSTEM_INTEGRATION_SUMMARY.md)