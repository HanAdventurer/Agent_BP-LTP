# DQN离线训练快速开始

## 📋 CSV数据集格式（必需的9列）

```csv
data_size,bit_error_rate,delay_ms,transmission_rate_mbps,bundle_size,ltp_block_size,ltp_segment_size,session_count,delivery_time_ms
10240,0.00001,50.0,10.0,2000,60000,400,6,850.5
20480,0.00005,100.0,8.0,4000,100000,600,8,1250.2
```

**字段说明**：
- 前4列：**输入**（网络状态）
- 中4列：**输出**（协议参数）
- 最后1列：**性能**（交付时间）

**约束条件**：`ltp_block_size >= bundle_size AND ltp_block_size % bundle_size == 0`

---

## 🚀 快速测试（30秒）

```bash
# 使用示例数据集测试
bash /root/agent/test_offline_training.sh
```

---

## 💡 基本用法

```bash
# 最简单的用法（使用示例数据集）
cd /root/agent/computer
python3 offline_training.py --dataset /root/agent/training_dataset_example.csv
```

---

## ⚙️ 常用选项

```bash
# 完整训练（10轮）
python3 offline_training.py \
    --dataset your_data.csv \
    --epochs 10 \
    --batch-size 50 \
    --save-path /path/to/model.pth

# 继续训练已有模型
python3 offline_training.py \
    --dataset your_data.csv \
    --epochs 5 \
    --load-model /path/to/existing_model.pth \
    --save-path /path/to/updated_model.pth

# 强制使用CPU（无GPU时）
python3 offline_training.py \
    --dataset your_data.csv \
    --cpu-only
```

---

## 📊 如何准备数据集

### 方法1：使用历史接收端数据
```bash
cp /root/agent/receive/receiver_records.csv /root/agent/my_dataset.csv
```

### 方法2：使用示例数据集
```bash
cp /root/agent/training_dataset_example.csv /root/agent/my_dataset.csv
# 编辑 my_dataset.csv 添加您的数据
```

### 方法3：手动创建
按照上面的CSV格式，在Excel或文本编辑器中创建数据。

---

## ✅ 数据集质量检查

```bash
# 检查记录数
tail -n +2 your_data.csv | wc -l

# 检查约束（Python）
python3 << 'EOF'
import csv
with open('your_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        bundle = int(float(row['bundle_size']))
        block = int(float(row['ltp_block_size']))
        if block < bundle or block % bundle != 0:
            print(f"❌ 行{i+2}: block={block}, bundle={bundle}")
EOF
```

---

## 📈 推荐训练参数

| 数据集大小 | epochs | batch_size | 预期时间 |
|-----------|--------|-----------|---------|
| < 100条 | 5-10 | 50 | 1-2分钟 |
| 100-500条 | 3-5 | 100 | 2-5分钟 |
| > 500条 | 1-3 | 100 | 5-10分钟 |

---

## 🎯 典型工作流

```bash
# 1. 准备数据集（50条以上）
vim my_training_data.csv

# 2. 离线预训练（5轮）
python3 offline_training.py \
    --dataset my_training_data.csv \
    --epochs 5 \
    --save-path model_pretrained.pth

# 3. 部署模型到在线优化器
# （需要修改 mode_dqn_v2_gpu.py 加载 model_pretrained.pth）

# 4. 启动在线服务（继续学习）
python3 mode_dqn_v2_gpu.py
```

---

## 🔍 验证训练效果

训练完成后，查看输出：

```
最终平均奖励: 0.3521  ← 如果 > 0.2，说明训练有效
最终探索率: 0.0773    ← 如果 < 0.05，说明接近收敛
```

**判断标准**：
- 平均奖励 > 0.2：✅ 模型可用
- 平均奖励 > 0.4：✅ 模型优秀
- 探索率 < 0.02：✅ 已收敛

---

## ❌ 常见问题

**Q: 训练时提示"CUDA out of memory"**
```bash
# 解决：使用CPU模式
python3 offline_training.py --dataset data.csv --cpu-only
```

**Q: CSV加载失败**
```bash
# 检查文件格式（必须是UTF-8编码的CSV）
file your_data.csv
head -2 your_data.csv  # 查看前2行
```

**Q: 约束验证失败**
```bash
# 确保 block % bundle == 0
# 例如：bundle=2000, block必须是 40000, 60000, 80000 等
```

---

## 📚 完整文档

详细说明请查看：[OFFLINE_TRAINING_GUIDE.md](OFFLINE_TRAINING_GUIDE.md)

---

## 📝 快速参考

**必需的CSV列**（按顺序）：
1. data_size
2. bit_error_rate
3. delay_ms
4. transmission_rate_mbps
5. bundle_size
6. ltp_block_size
7. ltp_segment_size
8. session_count
9. delivery_time_ms

**命令行参数**：
- `--dataset`: CSV文件路径（必需）
- `--epochs`: 训练轮数（默认5）
- `--batch-size`: 批次大小（默认50）
- `--save-path`: 模型保存路径
- `--load-model`: 加载已有模型
- `--cpu-only`: 强制CPU模式

---

**最后更新**: 2026-02-18
