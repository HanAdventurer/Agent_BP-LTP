# 快速启动指南

## 系统要求

### 硬件
- 2个Linux嵌入式板卡（节点A和B）
- 1个带GPU的Windows PC（优化器C）
- 网络连接（最好直连或同一网段）

### 软件
- Python 3.6+
- ION-DTN系统（用于BP/LTP，可选）
- Linux网络工具：tc, iproute2

## 文件清单

确保以下文件在 `/root/agent/` 目录：

```
✓ sender.py                      # 发送节点A
✓ receiver.py                    # 接收节点B
✓ mode.py                        # 优化器C
✓ bp_ltp_interface.py           # 发送端接口
✓ bp_ltp_receiver_interface.py  # 接收端接口
✓ dtn_ion.py                    # BP/LTP API
✓ network_config.csv            # 网络配置
```

## 部署步骤

### 第1步：网络配置

确定节点IP地址（示例）：
- 节点A（发送端）: 192.168.1.1
- 节点B（接收端）: 192.168.1.2
- 优化器C: 192.168.1.3

### 第2步：启动接收端（节点B）

在节点B的Linux板卡上执行：

```bash
cd /root/agent

# 纯模拟模式（推荐用于初始测试）
python3 receiver.py \
  --listen-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5003 \
  --simulate

# 或使用BP/LTP模式
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5003
```

**预期输出**：
```
============================================================
接收节点B启动
监听端口: 5001
优化器: 192.168.1.3:5003
============================================================
[监听] 在端口 5001 上监听来自节点A的连接...
[记录刷新线程] 启动
```

### 第3步：启动优化器（电脑C）

在Windows PC上执行：

```bash
cd \root\agent

# 启动优化器
python mode.py \
  --param-port 5002 \
  --record-port 5003
```

**预期输出**：
```
============================================================
优化器(算力电脑C)启动
参数请求服务器: 端口 5002
记录接收服务器: 端口 5003
============================================================
[参数请求服务器] 启动 (监听端口 5002)
[记录接收服务器] 启动 (监听端口 5003)
```

### 第4步：启动发送端（节点A）

在节点A的Linux板卡上执行：

```bash
cd /root/agent

# 纯模拟模式（推荐用于初始测试）
python3 sender.py \
  --receiver-host 192.168.1.2 \
  --receiver-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5002 \
  --config-file network_config.csv \
  --interval 10 \
  --simulate

# 或使用BP/LTP模式
python3 sender.py \
  --use-bp-ltp \
  --source-eid "ipn:1.1" \
  --destination-eid 2 \
  --receiver-host 192.168.1.2 \
  --receiver-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5002 \
  --config-file network_config.csv \
  --interval 10
```

**预期输出**：
```
============================================================
发送节点A启动
接收节点B: 192.168.1.2:5001
优化器C: 192.168.1.3:5002
============================================================
[配置加载] 成功从 network_config.csv 加载 10 条配置
[初始化] BP/LTP接口已启用
[监听] 在端口 5002 上监听...

============================================================
开始新的传输周期 (配置索引: 0)
============================================================
[业务请求] 待发送数据量: 10240 bytes (从CSV配置读取)
[CSV配置 1] Normal link condition
[链路状态] 误码率: 1e-05, 延时: 100.0ms, 速率: 10.0Mbps
[请求优化] 已发送请求到优化器 192.168.1.3:5002
[收到优化参数] {'bundle_size': 1024, ...}
...
```

## 监控运行

### 查看日志

```bash
# 实时查看发送端日志
tail -f /root/agent/sender.log

# 实时查看接收端日志
tail -f /root/agent/receiver.log

# 实时查看优化器日志
tail -f /root/agent/optimizer.log
```

### 检查系统状态

```bash
# 检查端口监听
netstat -tlnp | grep -E "5001|5002|5003"

# 检查进程运行
ps aux | grep python3

# 检查网络连接
ss -tlnp
```

## 验证运行

### 1. 检查参数流

发送端日志应显示：
```
[请求优化] 已发送请求到优化器
[收到优化参数] {...}
```

### 2. 检查数据传输

接收端日志应显示：
```
[新连接] 来自 192.168.1.1
[数据接收] 开始时间: ...
[元数据接收] 完成时间戳: ...
[传输指标] 业务交付时间: ...ms
```

### 3. 检查记录生成

接收端日志应显示：
```
[记录器] 添加新记录 (当前缓冲数: 1/100)
[记录器] 添加新记录 (当前缓冲数: 2/100)
```

### 4. 检查模型更新

优化器日志应显示：
```
[收到训练记录] N 条来自 192.168.1.2
[批量更新] 开始使用 N 条记录进行训练
[模型更新] 状态: ..., 动作: ..., 奖励: ...
[模型更新完成] 模型版本: 1
```

## 常见问题

### Q1: 连接超时

**症状**：
```
[错误] 请求优化参数失败: [Errno 111] Connection refused
```

**原因**：
- 优化器未启动
- IP地址或端口不正确

**解决**：
```bash
# 检查优化器是否运行
ps aux | grep mode.py

# 检查端口是否开放
netstat -tlnp | grep 5002

# 确认IP地址
ifconfig eth0  # 查看实际IP
```

### Q2: CSV文件找不到

**症状**：
```
[警告] 配置文件 network_config.csv 不存在，使用默认参数
```

**解决**：
```bash
# 检查文件位置
ls -la /root/agent/network_config.csv

# 如果不存在，复制示例
cp network_config.csv.example network_config.csv
```

### Q3: 权限不足

**症状**：
```
[错误] 配置网络参数失败: Permission denied
```

**解决**：
```bash
# 授予tc命令权限
sudo visudo
# 添加一行: your_user ALL=(ALL) NOPASSWD: /sbin/tc

# 或以sudo运行
sudo python3 sender.py --use-bp-ltp ...
```

### Q4: bpcounter找不到

**症状**：
```
[错误] 监听接收失败: No such file or directory
```

**解决**：
```bash
# 检查ION是否安装
which bpcounter

# 安装ION-DTN
# Ubuntu: sudo apt-get install ion
# 或从源码编译
```

## 调试模式

### 启用详细日志

```python
# 在sender.py中
import logging
logging.basicConfig(level=logging.DEBUG)

# 在receiver.py中
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 测试单个组件

```bash
# 测试CSV读取
python3 -c "import sender; s = sender.SenderNode(); print(s.config_data)"

# 测试网络连接
python3 -c "import socket; s = socket.socket(); s.connect(('192.168.1.3', 5002))"

# 测试BP/LTP接口
python3 -c "from bp_ltp_interface import BPLTPInterface; b = BPLTPInterface()"
```

## 性能测试

### 测试1: 单次传输

```bash
# 设置较长的间隔，观察单次传输过程
python3 sender.py \
  --simulate \
  --receiver-host 192.168.1.2 \
  --interval 60

# 观察日志
tail -f sender.log
tail -f receiver.log
```

### 测试2: 连续运行

```bash
# 运行足够长的时间观察优化过程
python3 sender.py \
  --simulate \
  --receiver-host 192.168.1.2 \
  --interval 5

# 观察模型是否不断改进
tail -f optimizer.log | grep "奖励"
```

### 测试3: 多场景循环

```bash
# 验证CSV循环使用
python3 -c "
import sender
s = sender.SenderNode()
for i in range(len(s.config_data) + 2):
    state = s.get_link_state()
    print(f'{i}: {state}')
    s.config_index = (s.config_index + 1) % len(s.config_data)
"
```

## 下一步

### 1. 自定义网络场景

编辑 `network_config.csv`：
```csv
sequence,data_size_bytes,bit_error_rate,delay_ms,transmission_rate_mbps,description
1,50000,5e-5,50.0,20.0,Ultra-fast link
2,100000,1e-3,500.0,1.0,Extreme deep-space
```

### 2. 调整强化学习参数

在 `mode.py` 中修改：
```python
self.learning_rate = 0.2  # 学习率
self.discount_factor = 0.95  # 折扣因子
self.epsilon = 0.05  # 探索率
```

### 3. 集成实际BP/LTP

将现有的ION命令替换为真实的BP/LTP调用，参考 `BP_LTP_INTEGRATION_GUIDE.md`。

### 4. 部署到生产环境

- 在真实的嵌入式板卡上运行
- 使用systemd/supervisord管理进程
- 配置日志轮转和备份
- 监控系统资源使用

## 支持

如遇问题，查看以下文档：

1. [BP_LTP_INTEGRATION_GUIDE.md](BP_LTP_INTEGRATION_GUIDE.md) - 发送端详解
2. [BP_LTP_RECEIVER_GUIDE.md](BP_LTP_RECEIVER_GUIDE.md) - 接收端详解
3. [SYSTEM_INTEGRATION_SUMMARY.md](SYSTEM_INTEGRATION_SUMMARY.md) - 系统概览

## 下一个传输周期

一旦系统正常运行，你将看到：

```
============================================================
开始新的传输周期 (配置索引: 0)
============================================================
[业务请求] 待发送数据量: 10240 bytes
[链路状态] 误码率: 1e-05, 延时: 100.0ms, 速率: 10.0Mbps
[请求优化] 已发送请求到优化器
[收到优化参数] {...}  # 参数在不断优化！
[参数应用] 已更新协议栈参数
[开始传输] 时间戳: 2026-02-10 12:34:56.123456
[传输完成] 接收节点确认: OK
[元数据发送] 已发送传输元数据到节点B

[周期完成] 传输周期成功完成
```

这表示系统工作正常！持续运行，优化器会学习到最优的参数组合。
