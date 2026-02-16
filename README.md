# BP/LTP自适应参数优化系统

## 📋 项目概述

这是一个基于强化学习的BP/LTP协议栈参数自适应优化系统。在深空通信/卫星通信背景下，根据实时的网络状态（误码率、延时、传输速率等），自动优化协议栈参数（Bundle大小、LTP Block大小、LTP Segment大小、会话数量），以最小化业务交付时间。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      BP/LTP优化系统                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  节点A(Linux板卡) ←→ 优化器C(Windows PC) ←→ 节点B(Linux板卡)  │
│   发送端              强化学习               接收端             │
│                                                                 │
│  功能:                功能:                  功能:             │
│  • 生成业务           • 参数优化              • 监听接收         │
│  • 获取链路状态       • 模型学习              • 时间测量         │
│  • 应用协议参数       • 反馈改进              • 记录生成         │
│  • 通过BP/LTP发送     • 持续优化              • 发送训练数据     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| **sender.py** | 发送节点A - 产生业务、请求参数、发送数据 |
| **receiver.py** | 接收节点B - 监听接收、测量时间、生成记录 |
| **mode.py** | 优化器C - 强化学习优化、模型更新 |
| **bp_ltp_interface.py** | 发送端BP/LTP接口封装 |
| **bp_ltp_receiver_interface.py** | 接收端BP/LTP接口封装 |
| **dtn_ion.py** | BP/LTP原始API（来自ION-DTN系统） |
| **network_config.csv** | 网络场景配置（10种场景） |
| **BP_LTP_INTEGRATION_GUIDE.md** | 发送端详细集成文档 |
| **BP_LTP_RECEIVER_GUIDE.md** | 接收端详细集成文档 |
| **SYSTEM_INTEGRATION_SUMMARY.md** | 完整系统架构和数据流 |
| **QUICK_START.md** | 快速启动指南 |
| **README.md** | 本文件 |

## 🚀 快速开始

### 最简单的方式（纯模拟，无需ION系统）

```bash
# 终端1：启动接收端
python3 receiver.py --simulate

# 终端2：启动优化器
python3 mode.py

# 终端3：启动发送端
python3 sender.py --simulate --receiver-host 192.168.1.2
```

### 完整的BP/LTP部署

```bash
# 终端1：启动接收端（需要ION系统）
python3 receiver.py --use-bp-ltp --own-eid-number 2

# 终端2：启动优化器
python3 mode.py

# 终端3：启动发送端（需要ION系统）
python3 sender.py --use-bp-ltp --receiver-host 192.168.1.2
```

详见 [QUICK_START.md](QUICK_START.md)

## 🔄 工作流程

### 单次传输周期

```
1. [发送端A] 从CSV读取业务大小和链路状态
        ↓
2. [发送端A] 向优化器C请求最优参数
        ↓
3. [优化器C] 基于RL生成优化的协议参数
        ↓
4. [发送端A] 应用参数并配置BP/LTP协议栈
        ↓
5. [发送端A] 通过BP/LTP发送数据（记录开始时间戳）
        ↓
6. [接收端B] 启动bpcounter监听接收
        ↓
7. [接收端B] 接收完成，记录完成时间戳
        ↓
8. [接收端B] 计算业务交付时间
        ↓
9. [接收端B] 生成训练记录（input + output + performance）
        ↓
10. [接收端B] 周期性发送训练记录到优化器C
        ↓
11. [优化器C] 使用训练数据更新强化学习模型
        ↓
12. [循环回到步骤1，参数不断优化！]
```

## 📊 核心功能

### 发送端功能

- ✅ **CSV配置管理** - 模拟10种不同网络场景
- ✅ **参数优化请求** - 向优化器请求最优参数
- ✅ **链路配置** - 配置网络带宽、延时、丢包率
- ✅ **协议配置** - 配置LTP span、bundle大小、session数
- ✅ **BP/LTP传输** - 使用send_bpdriver_command发送数据
- ✅ **元数据发送** - 发送传输参数给接收端

### 接收端功能

- ✅ **bpcounter监听** - 使用run_bpcounter_and_monitor监听接收
- ✅ **时间测量** - 记录开始时间和完成时间
- ✅ **交付时间计算** - 计算业务交付时间
- ✅ **记录生成** - 生成input/output/performance记录
- ✅ **批量发送** - 周期性或缓冲满后发送记录

### 优化器功能

- ✅ **状态离散化** - 将连续状态转换为离散状态
- ✅ **动作选择** - ε-贪心策略选择协议参数
- ✅ **Q-Learning更新** - 使用训练数据更新Q表
- ✅ **模型版本控制** - 跟踪模型演进
- ✅ **参数优化** - 不断改进协议参数选择

## 📈 API函数映射

### 发送端BP/LTP API

| dtn_ion函数 | 接口方法 | 功能 |
|-----------|---------|------|
| `configure_network()` | `configure_link_parameters()` | 配置网络参数 |
| `calculate_packet_loss()` | 内部调用 | 计算丢包率 |
| `calculate_ltp_sessions()` | `apply_protocol_parameters()` | 计算会话数 |
| `setup_ltp_span()` | `apply_protocol_parameters()` | 配置LTP span |
| `setup_contact()` | `setup_transmission_contact()` | 设置contact |
| `send_bpdriver_command()` | `transmit_data_via_bp_ltp()` | 发送数据 |

### 接收端BP/LTP API

| dtn_ion函数 | 接口方法 | 功能 |
|-----------|---------|------|
| `run_bpcounter_and_monitor()` | `monitor_reception()` | 监听接收 |
| - | `calculate_bundle_count()` | 计算bundle数 |
| - | `parse_bpcounter_report()` | 解析报告 |

## 🎯 关键参数

### 协议参数

```
bundle_size         # Bundle大小（bytes）：512, 1024, 2048, 4096
ltp_block_size      # LTP Block大小（bytes）：256, 512, 1024, 2048
ltp_segment_size    # LTP Segment大小（bytes）：128, 256, 512, 1024
session_count       # LTP会话数量：1, 2, 4, 8
```

### 网络参数

```
bit_error_rate      # 误码率：1e-5 ~ 1e-3
delay_ms            # 延时（毫秒）：50 ~ 500
transmission_rate_mbps  # 传输速率（Mbps）：1 ~ 20
data_size           # 数据大小（bytes）：5KB ~ 100KB
```

### 性能指标

```
delivery_time_ms    # 业务交付时间（毫秒）
```

## 🔧 高级配置

### 编辑CSV配置文件

```csv
sequence,data_size_bytes,bit_error_rate,delay_ms,transmission_rate_mbps,description
1,10240,1e-5,100.0,10.0,Normal link
2,20480,1e-4,150.0,8.0,Degraded link
3,5120,1e-3,200.0,5.0,Poor link
```

### 调整强化学习参数

编辑 `mode.py` 中的 `SimpleRLOptimizer` 类：

```python
self.learning_rate = 0.1        # 学习率
self.discount_factor = 0.9      # 折扣因子
self.epsilon = 0.1              # ε-贪心探索率
self.state_bins = [10, 10, 10, 10]  # 状态离散化级数
```

### 自定义网络配置

在 `sender.py` 或 `mode.py` 中：

```python
# 修改tc命令参数
execute_command(f"sudo tc qdisc add ... netem loss {loss_rate}% delay {delay}ms ...")
```

## 📚 文档

| 文档 | 内容 |
|------|------|
| [QUICK_START.md](QUICK_START.md) | 30分钟快速入门指南 |
| [BP_LTP_INTEGRATION_GUIDE.md](BP_LTP_INTEGRATION_GUIDE.md) | 发送端详细文档 |
| [BP_LTP_RECEIVER_GUIDE.md](BP_LTP_RECEIVER_GUIDE.md) | 接收端详细文档 |
| [SYSTEM_INTEGRATION_SUMMARY.md](SYSTEM_INTEGRATION_SUMMARY.md) | 完整系统架构 |

## 🧪 测试和验证

### 单元测试

```bash
# 测试CSV读取
python3 -c "from sender import SenderNode; s = SenderNode(); print(len(s.config_data))"

# 测试参数优化
python3 -c "from mode import SimpleRLOptimizer; o = SimpleRLOptimizer(); print(o.optimize_params({}))"

# 测试时间计算
python3 -c "from receiver import RecordLogger; r = RecordLogger(); r.record_transmission(10240, 1e-5, 100, 10, 1024, 512, 256, 4, 1234.5)"
```

### 集成测试

```bash
# 启动完整系统
python3 receiver.py --simulate &
python3 mode.py &
python3 sender.py --simulate --receiver-host 127.0.0.1 &

# 观察运行
sleep 30
ps aux | grep python3
```

### 性能验证

```bash
# 查看优化过程
tail -f optimizer.log | grep "模型版本\|奖励"

# 查看交付时间变化
tail -f receiver.log | grep "业务交付时间"

# 统计参数使用频率
grep "优化参数\|收到优化参数" sender.log | sort | uniq -c
```

## 🐛 故障排除

常见问题的解决方案，见 [QUICK_START.md#常见问题](QUICK_START.md#常见问题)

## 📋 系统要求

### 最小配置（纯模拟）

- Python 3.6+
- Linux/Mac/Windows
- 内存：512MB
- 存储：10MB

### 完整配置（BP/LTP）

- ION-DTN系统
- Linux 3.10+
- 网络工具：iproute2, tc
- sudo权限（用于网络配置）
- 内存：1GB
- 存储：100MB

## 📦 依赖库

```python
# 标准库（已包含）
socket, json, time, struct, csv, os, math, threading, subprocess, re, datetime

# 可选库（仅用于高级功能）
numpy  # 用于大规模数据处理
pandas  # 用于数据分析
matplotlib  # 用于可视化
```

## 🎓 学习资源

### BP/LTP概念

- Bundle Protocol (RFC 5050)
- Licklider Transmission Protocol (RFC 5326)
- Delay-Tolerant Networking (DTN)

### 强化学习

- Q-Learning算法
- 马尔可夫决策过程 (MDP)
- 值函数和策略优化

### ION系统

- [ION官方文档](https://sourceforge.net/projects/ion-dtn/)
- [ION管理指南](https://sourceforge.net/p/ion-dtn/code/HEAD/tree/tags/rel-4-1-0/doc/)

## 💡 使用场景

### 1. 学术研究

- 协议优化算法研究
- DTN网络性能评估
- 强化学习应用研究

### 2. 系统测试

- 新参数组合验证
- 网络场景模拟
- 性能基准测试

### 3. 生产部署

- 实时参数优化
- 自适应传输
- 性能监控

## 🚀 下一步

1. **快速开始** - 按照 [QUICK_START.md](QUICK_START.md) 部署系统
2. **深入学习** - 阅读相应的集成文档
3. **自定义配置** - 根据实际需求调整参数
4. **生产部署** - 集成到实际的BP/LTP系统

## 📝 文件更新日志

### v1.0 (2026-02-10)

- ✅ 完整的三节点系统架构
- ✅ CSV配置管理
- ✅ BP/LTP协议栈集成
- ✅ 强化学习优化
- ✅ 完整的文档和示例

## 📞 支持

遇到问题？

1. 查看 [QUICK_START.md#常见问题](QUICK_START.md#常见问题)
2. 查看对应的详细文档
3. 检查日志输出
4. 验证网络连接和配置

## 📄 许可证

这是一个教学和研究项目。基于ION-DTN系统（NASA开源）和通用开源库。

## 🙏 致谢

- ION-DTN社区（BP/LTP协议实现）
- Python社区（标准库和工具）
- DTN研究社区（协议规范和最佳实践）

---

**准备好开始了吗？** → 访问 [QUICK_START.md](QUICK_START.md)
