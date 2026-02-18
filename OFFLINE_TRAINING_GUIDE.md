# DQN离线训练指南

**版本**: DQN v2.3 GPU
**日期**: 2026-02-17

---

## 概述

离线训练允许您使用历史数据集预先训练DQN模型，无需等待在线数据积累。这可以：
- ✅ 快速启动模型（预训练）
- ✅ 利用历史最佳实践数据
- ✅ 在部署前验证模型效果
- ✅ 加速模型收敛

---

## 1. 数据集格式

### 1.1 CSV文件格式

您的训练数据集应该是CSV格式，包含以下列：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| **data_size** | int | 数据量大小（bytes） | 10240 |
| **bit_error_rate** | float | 误码率 | 0.00001 |
| **delay_ms** | float | 延时（毫秒） | 100.0 |
| **transmission_rate_mbps** | float | 传输速率（Mbps） | 10.0 |
| **bundle_size** | int | Bundle大小（bytes） | 2000 |
| **ltp_block_size** | int | LTP Block大小（bytes） | 60000 |
| **ltp_segment_size** | int | LTP Segment大小（bytes） | 400 |
| **session_count** | int | 会话数量 | 6 |
| **delivery_time_ms** | float | 业务交付时间（毫秒） | 850.5 |
| throughput_mbps | float | 吞吐量（Mbps，可选） | 0.0965 |
| timestamp | float | 时间戳（可选） | 1708185600.123 |
| description | string | 描述（可选） | "良好网络条件" |

**注意**：
- 前9列（data_size到delivery_time_ms）是**必需的**
- 后3列（throughput_mbps、timestamp、description）是可选的
- Bundle和Block必须满足约束：`ltp_block_size >= bundle_size AND ltp_block_size % bundle_size == 0`

### 1.2 示例CSV文件

已为您创建了示例数据集：[training_dataset_example.csv](training_dataset_example.csv)

包含50条训练记录，覆盖多种网络场景：
- 良好网络（低延时、低误码率）
- 中等网络（中等延时、中等误码率）
- 恶劣网络（高延时、高误码率）
- 不同数据量大小（10KB ~ 100KB）

**CSV文件头部示例**：
```csv
data_size,bit_error_rate,delay_ms,transmission_rate_mbps,bundle_size,ltp_block_size,ltp_segment_size,session_count,delivery_time_ms,throughput_mbps,timestamp,description
10240,0.00001,50.0,10.0,2000,40000,200,5,850.5,0.0965,1708185600.123,"良好网络条件"
20480,0.00005,100.0,8.0,4000,80000,400,8,1250.2,0.1311,1708185660.456,"中等网络条件"
40960,0.0001,150.0,6.0,6000,120000,600,10,2100.8,0.1561,1708185720.789,"较差网络条件"
```

---

## 2. 准备训练数据集

### 2.1 从接收端CSV转换

如果您已经有接收端的记录CSV（receiver_records.csv），可以直接使用：

```bash
# 接收端CSV格式已经匹配训练数据集格式
cp /root/agent/receive/receiver_records.csv /root/agent/training_dataset.csv
```

### 2.2 手动创建数据集

您可以基于专家经验手动创建训练数据：

```csv
data_size,bit_error_rate,delay_ms,transmission_rate_mbps,bundle_size,ltp_block_size,ltp_segment_size,session_count,delivery_time_ms
10240,0.00001,50.0,10.0,2000,60000,400,6,750.0
20480,0.00005,100.0,8.0,4000,100000,600,8,1200.0
40960,0.0001,150.0,6.0,8000,160000,800,12,2000.0
```

**设计建议**：
1. 覆盖多种网络条件（好、中、差）
2. 包含不同数据量大小
3. 使用已知有效的参数组合
4. 至少50-100条记录

### 2.3 从实验数据导出

如果您有实际测试数据，可以编写脚本转换为CSV格式：

```python
#!/usr/bin/env python3
import csv

# 您的实验数据
experiment_data = [
    {
        'input': {'data_size': 10240, 'bit_error_rate': 0.00001, ...},
        'output': {'bundle_size': 2000, 'ltp_block_size': 60000, ...},
        'performance': {'delivery_time_ms': 850.5}
    },
    # ... 更多数据
]

# 转换为CSV
with open('training_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'data_size', 'bit_error_rate', 'delay_ms', 'transmission_rate_mbps',
        'bundle_size', 'ltp_block_size', 'ltp_segment_size', 'session_count',
        'delivery_time_ms'
    ])

    for data in experiment_data:
        writer.writerow([
            data['input']['data_size'],
            data['input']['bit_error_rate'],
            data['input']['delay_ms'],
            data['input']['transmission_rate_mbps'],
            data['output']['bundle_size'],
            data['output']['ltp_block_size'],
            data['output']['ltp_segment_size'],
            data['output']['session_count'],
            data['performance']['delivery_time_ms']
        ])
```

---

## 3. 使用离线训练

### 3.1 基本用法

```bash
# 使用示例数据集训练（GPU模式）
cd /root/agent/computer
python3 offline_training.py --dataset /root/agent/training_dataset_example.csv
```

**输出示例**：
```
[初始化] 使用GPU模式进行离线训练
[GPU] 检测到CUDA设备: NVIDIA GeForce RTX 4060
[配置] 模型保存路径: /root/agent/computer/dqn_model_pretrained.pth
[数据加载] 成功从 /root/agent/training_dataset_example.csv 加载 50 条训练记录
[验证] 交付时间范围: 720.10 ~ 6200.80 ms
[验证] ✅ 数据集验证通过

======================================================================
开始离线训练
======================================================================
训练配置:
  • 数据集大小: 50 条记录
  • 训练轮数: 5
  • 批次大小: 50
  • 每轮批次数: 1
  • 模式: GPU
======================================================================

[Epoch 1/5] 开始训练
[DQN训练-GPU] 开始使用 50 条记录进行批量训练
...
[Epoch 1/5] 完成
  • 当前探索率: 0.0950
  • 模型版本: 1
  • 平均奖励: 0.1234

...

======================================================================
离线训练完成
======================================================================
训练统计:
  • 总训练批次: 5
  • 总耗时: 15.23 秒
  • 平均每批: 3.05 秒
  • 最终模型版本: 5
  • 最终探索率: 0.0773
  • 最终平均奖励: 0.3521
======================================================================

[保存] ✅ GPU模型已保存到: /root/agent/computer/dqn_model_pretrained.pth

✅ 训练完成！模型已保存到: /root/agent/computer/dqn_model_pretrained.pth
```

### 3.2 高级选项

#### 指定训练轮数
```bash
# 训练10轮（完整遍历数据集10次）
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --epochs 10
```

#### 调整批次大小
```bash
# 每批处理30条记录
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --batch-size 30
```

#### 定期保存中间模型
```bash
# 每2批保存一次
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --epochs 10 \
    --save-interval 2
```

#### 自定义保存路径
```bash
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --save-path /root/agent/models/my_model.pth
```

#### 加载已有模型继续训练
```bash
# 先训练5轮
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --epochs 5 \
    --save-path /root/agent/models/model_v1.pth

# 加载后继续训练5轮
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --epochs 5 \
    --load-model /root/agent/models/model_v1.pth \
    --save-path /root/agent/models/model_v2.pth
```

#### 强制使用CPU模式
```bash
# 在没有GPU或测试CPU性能时使用
python3 offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --cpu-only
```

---

## 4. 在线服务中使用预训练模型

### 4.1 修改GPU优化器加载预训练模型

编辑 [mode_dqn_v2_gpu.py](computer/mode_dqn_v2_gpu.py)，在 `DQNOptimizerGPU.__init__()` 中添加加载逻辑：

```python
def __init__(self, device=None, pretrained_model: str = None):
    """初始化DQN优化器（GPU版本）"""
    # ... 现有初始化代码 ...

    # 加载预训练模型（如果提供）
    if pretrained_model and os.path.exists(pretrained_model):
        self.load_pretrained_model(pretrained_model)

def load_pretrained_model(self, model_path: str):
    """加载预训练模型"""
    try:
        checkpoint = torch.load(model_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_version = checkpoint.get('model_version', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        self.epsilon = checkpoint.get('epsilon', 0.1)

        if 'episode_rewards' in checkpoint:
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)

        # 更新推理快照
        self._create_inference_snapshot()

        print(f"[预训练] ✅ 已加载预训练模型: {model_path}")
        print(f"  • 模型版本: {self.model_version}")
        print(f"  • 探索率: {self.epsilon:.4f}")

    except Exception as e:
        print(f"[警告] 加载预训练模型失败: {e}")
```

### 4.2 启动优化器时指定预训练模型

```bash
# 修改 mode_dqn_v2_gpu.py 的 main() 函数
cd /root/agent/computer

# 添加命令行参数
python3 mode_dqn_v2_gpu.py --pretrained-model /root/agent/computer/dqn_model_pretrained.pth
```

**或者直接在代码中硬编码**：

```python
# mode_dqn_v2_gpu.py 的 OptimizerServer.__init__()
self.dqn_optimizer = DQNOptimizerGPU(
    pretrained_model="/root/agent/computer/dqn_model_pretrained.pth"
)
```

---

## 5. 评估预训练模型

### 5.1 使用评估脚本

```bash
# 先用离线数据训练
python3 /root/agent/computer/offline_training.py \
    --dataset /root/agent/training_dataset_example.csv \
    --epochs 10

# 然后评估模型（需要在线运行一段时间后评估）
# 这里只能通过日志评估训练过程
```

### 5.2 查看训练统计

离线训练完成后，会输出：
- 最终平均奖励
- 最终探索率
- 训练步数

**判断标准**：
- 平均奖励 > 0.2：模型已基本收敛
- 探索率 ≤ 0.05：仍在探索，可继续训练
- 探索率 ≤ 0.02：接近最优策略

---

## 6. 最佳实践

### 6.1 数据集准备

1. **数据量**：
   - 最少50条记录
   - 推荐100-200条
   - 更多数据 = 更好效果

2. **数据多样性**：
   - 覆盖不同网络条件（好、中、差）
   - 覆盖不同数据量大小
   - 包含边缘情况（极端高/低延时）

3. **数据质量**：
   - 确保参数组合满足约束
   - 确保delivery_time准确
   - 移除异常数据

### 6.2 训练策略

1. **分阶段训练**：
   ```bash
   # 阶段1：预训练（5轮）
   python3 offline_training.py \
       --dataset training_data.csv \
       --epochs 5 \
       --save-path model_stage1.pth

   # 阶段2：精调（10轮，更小探索率）
   python3 offline_training.py \
       --dataset training_data.csv \
       --epochs 10 \
       --load-model model_stage1.pth \
       --save-path model_stage2.pth

   # 阶段3：在线训练
   # 使用model_stage2.pth启动在线优化器
   ```

2. **批次大小选择**：
   - 数据集 < 100条：batch_size = 全部数据
   - 数据集 100-500条：batch_size = 50
   - 数据集 > 500条：batch_size = 100

3. **训练轮数选择**：
   - 小数据集（< 100条）：5-10轮
   - 中型数据集（100-500条）：3-5轮
   - 大数据集（> 500条）：1-3轮

### 6.3 混合训练策略（推荐）

```bash
# 步骤1：离线预训练（快速启动）
python3 offline_training.py \
    --dataset historical_data.csv \
    --epochs 5 \
    --save-path model_pretrained.pth

# 步骤2：启动在线优化器（加载预训练模型）
# 修改 mode_dqn_v2_gpu.py 加载 model_pretrained.pth

# 步骤3：在线持续学习
# 优化器会继续从在线数据学习并改进
```

**优势**：
- ✅ 快速达到基本性能（离线预训练）
- ✅ 持续适应实际网络（在线学习）
- ✅ 最佳效果

---

## 7. 故障排除

### 问题1：数据加载失败

**错误**：`[错误] CSV文件不存在`

**解决**：
```bash
# 检查文件路径
ls -lh /root/agent/training_dataset_example.csv

# 使用绝对路径
python3 offline_training.py --dataset /root/agent/training_dataset_example.csv
```

### 问题2：约束验证失败

**错误**：`[验证] ⚠️  记录X: block(40000) % bundle(6000) != 0`

**解决**：检查CSV数据，确保 `ltp_block_size % bundle_size == 0`

### 问题3：GPU内存不足

**错误**：`CUDA out of memory`

**解决方案**：
```bash
# 方案1：使用CPU模式
python3 offline_training.py --dataset data.csv --cpu-only

# 方案2：减小批次大小
python3 offline_training.py --dataset data.csv --batch-size 20

# 方案3：清理GPU缓存
import torch
torch.cuda.empty_cache()
```

### 问题4：训练速度慢

**原因**：数据集太大或批次太小

**优化**：
```bash
# 增大批次大小（如果GPU内存允许）
python3 offline_training.py --dataset data.csv --batch-size 100

# 减少训练轮数
python3 offline_training.py --dataset data.csv --epochs 3
```

---

## 8. 完整示例

### 端到端离线训练流程

```bash
#!/bin/bash
# 完整的离线训练示例

echo "步骤1: 准备数据集"
cp /root/agent/training_dataset_example.csv /root/agent/my_training_data.csv

echo "步骤2: 验证数据集"
head -5 /root/agent/my_training_data.csv

echo "步骤3: 离线训练（10轮）"
cd /root/agent/computer
python3 offline_training.py \
    --dataset /root/agent/my_training_data.csv \
    --epochs 10 \
    --batch-size 50 \
    --save-path /root/agent/computer/dqn_model_v1.pth

echo "步骤4: 检查训练结果"
ls -lh /root/agent/computer/dqn_model_v1.pth

echo "步骤5: 部署模型（需要修改 mode_dqn_v2_gpu.py 加载模型）"
echo "完成！模型已准备好用于在线服务"
```

---

## 9. 总结

### 离线训练的优势

| 优势 | 说明 |
|------|------|
| 🚀 **快速启动** | 无需等待在线数据积累 |
| 📚 **利用历史数据** | 充分利用过往经验 |
| 🎯 **可控训练** | 可以精确控制训练过程 |
| 🧪 **离线验证** | 在部署前验证模型 |
| 💰 **成本低** | 无需实际传输数据 |

### 与在线训练对比

| 特性 | 离线训练 | 在线训练 |
|------|---------|---------|
| 数据来源 | 历史CSV | 实时传输 |
| 训练速度 | 快（无等待） | 慢（需积累） |
| 数据质量 | 可控 | 可能有噪声 |
| 适应性 | 固定 | 动态适应 |
| 推荐用途 | 预训练/冷启动 | 持续优化 |

### 推荐工作流

```
1. 离线预训练（5-10轮）
   ↓
2. 部署到在线服务
   ↓
3. 在线持续学习
   ↓
4. 定期评估性能
   ↓
5. 收集新数据 → 回到步骤1（周期性）
```

---

**编写者**: Claude Opus 4.6
**更新日期**: 2026-02-17
**适用版本**: DQN v2.3-GPU