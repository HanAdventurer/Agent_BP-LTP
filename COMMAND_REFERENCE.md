# 命令参考速查表

## 发送节点A (sender.py)

### 基本命令

```bash
python3 sender.py [OPTIONS]
```

### 必需参数

无（所有参数都有默认值）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--receiver-host` | string | 192.168.1.2 | 接收节点B的IP地址 |
| `--receiver-port` | int | 5001 | 接收节点B的端口 |
| `--optimizer-host` | string | 192.168.1.3 | 优化器C的IP地址 |
| `--optimizer-port` | int | 5002 | 优化器C的端口 |
| `--config-file` | string | network_config.csv | 网络配置CSV文件路径 |
| `--interval` | int | 60 | 传输周期间隔（秒） |
| `--source-eid` | string | ipn:1.1 | 源节点EID |
| `--destination-eid` | int | 2 | 目标节点EID数字 |
| `--dest-udp-addr` | string | 192.168.1.2:1113 | 目标UDP地址 |
| `--use-bp-ltp` | flag | False | 启用BP/LTP协议栈 |
| `--simulate` | flag | False | 强制使用模拟模式 |

### 使用示例

#### 1. 纯模拟模式（最简单）
```bash
python3 sender.py --simulate
```

#### 2. 自定义网络
```bash
python3 sender.py \
  --receiver-host 10.0.0.2 \
  --optimizer-host 10.0.0.3 \
  --simulate
```

#### 3. 使用BP/LTP
```bash
python3 sender.py \
  --use-bp-ltp \
  --source-eid "ipn:1.1" \
  --destination-eid 2 \
  --receiver-host 192.168.1.2
```

#### 4. 自定义CSV和间隔
```bash
python3 sender.py \
  --config-file my_scenarios.csv \
  --interval 10 \
  --simulate
```

#### 5. 完整的生产配置
```bash
python3 sender.py \
  --use-bp-ltp \
  --source-eid "ipn:1.1" \
  --destination-eid 2 \
  --receiver-host 192.168.100.10 \
  --receiver-port 5001 \
  --optimizer-host 192.168.100.20 \
  --optimizer-port 5002 \
  --dest-udp-addr "192.168.100.10:1113" \
  --config-file /path/to/config.csv \
  --interval 30
```

---

## 接收节点B (receiver.py)

### 基本命令

```bash
python3 receiver.py [OPTIONS]
```

### 必需参数

无（所有参数都有默认值）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--listen-port` | int | 5001 | 监听端口 |
| `--optimizer-host` | string | 192.168.1.3 | 优化器C的IP地址 |
| `--optimizer-port` | int | 5003 | 优化器C的数据接收端口 |
| `--own-eid-number` | int | 2 | 本节点EID数字 |
| `--use-bp-ltp` | flag | False | 启用BP/LTP协议栈接收 |
| `--simulate` | flag | False | 强制使用模拟模式 |

### 使用示例

#### 1. 纯模拟模式（最简单）
```bash
python3 receiver.py --simulate
```

#### 2. 自定义端口
```bash
python3 receiver.py \
  --listen-port 6001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 6003 \
  --simulate
```

#### 3. 使用BP/LTP
```bash
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001
```

#### 4. 自定义EID
```bash
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 5 \
  --optimizer-host 192.168.1.3
```

#### 5. 完整的生产配置
```bash
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001 \
  --optimizer-host 192.168.100.20 \
  --optimizer-port 5003
```

---

## 优化器C (mode.py)

### 基本命令

```bash
python mode.py [OPTIONS]
```

### 必需参数

无（所有参数都有默认值）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--param-port` | int | 5002 | 接收参数请求的端口 |
| `--record-port` | int | 5003 | 接收训练记录的端口 |

### 使用示例

#### 1. 默认端口（最简单）
```bash
python mode.py
```

#### 2. 自定义端口
```bash
python mode.py \
  --param-port 6002 \
  --record-port 6003
```

#### 3. 生产配置
```bash
python mode.py \
  --param-port 5002 \
  --record-port 5003
```

---

## 典型部署场景

### 场景1：本地测试（单机三进程）

```bash
# 终端1: 接收端
python3 receiver.py \
  --simulate \
  --listen-port 5001 \
  --optimizer-host 127.0.0.1 \
  --optimizer-port 5003

# 终端2: 优化器
python mode.py \
  --param-port 5002 \
  --record-port 5003

# 终端3: 发送端
python3 sender.py \
  --simulate \
  --receiver-host 127.0.0.1 \
  --receiver-port 5001 \
  --optimizer-host 127.0.0.1 \
  --optimizer-port 5002 \
  --interval 10
```

### 场景2：局域网部署（三台设备）

假设：
- 节点A: 192.168.1.1
- 节点B: 192.168.1.2
- 优化器C: 192.168.1.3

```bash
# 在节点B上
python3 receiver.py \
  --simulate \
  --listen-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5003

# 在优化器C上
python mode.py \
  --param-port 5002 \
  --record-port 5003

# 在节点A上
python3 sender.py \
  --simulate \
  --receiver-host 192.168.1.2 \
  --receiver-port 5001 \
  --optimizer-host 192.168.1.3 \
  --optimizer-port 5002 \
  --interval 20
```

### 场景3：生产环境（带BP/LTP）

```bash
# 在节点B上（Linux板卡）
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001 \
  --optimizer-host 192.168.100.20 \
  --optimizer-port 5003

# 在优化器C上（Windows PC）
python mode.py \
  --param-port 5002 \
  --record-port 5003

# 在节点A上（Linux板卡）
python3 sender.py \
  --use-bp-ltp \
  --source-eid "ipn:1.1" \
  --destination-eid 2 \
  --receiver-host 192.168.100.10 \
  --receiver-port 5001 \
  --optimizer-host 192.168.100.20 \
  --optimizer-port 5002 \
  --dest-udp-addr "192.168.100.10:1113" \
  --config-file network_config.csv \
  --interval 30
```

---

## 后台运行

### 使用nohup

```bash
# 接收端
nohup python3 receiver.py --simulate > receiver.log 2>&1 &

# 优化器
nohup python mode.py > optimizer.log 2>&1 &

# 发送端
nohup python3 sender.py --simulate > sender.log 2>&1 &
```

### 使用screen

```bash
# 启动screen会话
screen -S receiver
python3 receiver.py --simulate
# Ctrl+A, D 分离会话

screen -S optimizer
python mode.py
# Ctrl+A, D

screen -S sender
python3 sender.py --simulate
# Ctrl+A, D

# 重新连接会话
screen -r receiver
screen -r optimizer
screen -r sender
```

### 使用systemd

创建服务文件 `/etc/systemd/system/bp-ltp-receiver.service`:

```ini
[Unit]
Description=BP/LTP Receiver Node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/agent
ExecStart=/usr/bin/python3 /root/agent/receiver.py --use-bp-ltp
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用和启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable bp-ltp-receiver
sudo systemctl start bp-ltp-receiver
sudo systemctl status bp-ltp-receiver
```

---

## 调试命令

### 检查进程

```bash
# 查看Python进程
ps aux | grep python3

# 查看端口监听
netstat -tlnp | grep -E "5001|5002|5003"
ss -tlnp | grep -E "5001|5002|5003"

# 查看网络连接
netstat -anp | grep -E "5001|5002|5003"
ss -anp | grep -E "5001|5002|5003"
```

### 检查日志

```bash
# 实时查看日志
tail -f sender.log
tail -f receiver.log
tail -f optimizer.log

# 查看最近100行
tail -n 100 sender.log

# 搜索错误
grep -i error sender.log
grep -i warning receiver.log
```

### 测试网络连接

```bash
# 测试端口是否可达
telnet 192.168.1.3 5002
nc -zv 192.168.1.3 5002

# 测试HTTP连接
curl http://192.168.1.3:5002

# ping测试
ping 192.168.1.3
```

### 手动测试组件

```bash
# 测试CSV读取
python3 -c "
from sender import SenderNode
s = SenderNode()
print('配置数量:', len(s.config_data))
print('第一条配置:', s.config_data[0] if s.config_data else '无')
"

# 测试优化器
python3 -c "
from mode import SimpleRLOptimizer
optimizer = SimpleRLOptimizer()
state = {'data_size': 10240, 'bit_error_rate': 1e-5, 'delay_ms': 100, 'transmission_rate_mbps': 10}
params = optimizer.optimize_params(state)
print('优化参数:', params)
"

# 测试记录器
python3 -c "
from receiver import RecordLogger
logger = RecordLogger()
logger.record_transmission(10240, 1e-5, 100, 10, 1024, 512, 256, 4, 1234.5)
print('记录数:', len(logger.records))
"
```

---

## 停止和清理

### 停止进程

```bash
# 找到进程ID
ps aux | grep python3 | grep -E "sender|receiver|mode"

# 停止单个进程
kill <PID>

# 强制停止
kill -9 <PID>

# 停止所有Python进程（谨慎使用）
pkill -f "python3.*sender.py"
pkill -f "python3.*receiver.py"
pkill -f "python.*mode.py"
```

### 清理日志

```bash
# 清空日志文件
> sender.log
> receiver.log
> optimizer.log

# 删除日志文件
rm -f sender.log receiver.log optimizer.log

# 归档日志
tar -czf logs-$(date +%Y%m%d).tar.gz *.log
```

### 重置系统

```bash
# 停止所有进程
pkill -f "python.*sender.py"
pkill -f "python.*receiver.py"
pkill -f "python.*mode.py"

# 清理日志
rm -f *.log

# 清理网络配置（如果使用了tc）
sudo tc qdisc del dev eth0 root

# 重启ION系统（如果使用BP/LTP）
ionstop
ionstart
```

---

## 环境变量

### 设置Python路径

```bash
export PYTHONPATH=/root/agent:$PYTHONPATH
```

### 设置ION路径

```bash
export ION_HOME=/usr/local/ion
export PATH=$ION_HOME/bin:$PATH
```

### 完整的环境设置

创建 `setup_env.sh`:

```bash
#!/bin/bash
export PYTHONPATH=/root/agent:$PYTHONPATH
export ION_HOME=/usr/local/ion
export PATH=$ION_HOME/bin:$PATH

echo "环境设置完成"
echo "PYTHONPATH: $PYTHONPATH"
echo "ION_HOME: $ION_HOME"
```

使用：

```bash
source setup_env.sh
```

---

## 快速命令备忘录

### 最小启动（本地测试）

```bash
python3 receiver.py --simulate &
python mode.py &
python3 sender.py --simulate --receiver-host 127.0.0.1 --interval 10
```

### 生产启动（BP/LTP）

```bash
# 节点B
python3 receiver.py --use-bp-ltp --own-eid-number 2

# 优化器C
python mode.py

# 节点A
python3 sender.py --use-bp-ltp --receiver-host 192.168.1.2
```

### 查看状态

```bash
ps aux | grep python3
netstat -tlnp | grep -E "5001|5002|5003"
tail -f *.log
```

### 停止所有

```bash
pkill -f "python.*py"
```

---

需要更多帮助？查看 [README.md](README.md) 或 [QUICK_START.md](QUICK_START.md)
