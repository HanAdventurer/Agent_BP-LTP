# BP/LTP协议栈集成指南

## 概述

本系统已经将BP/LTP协议栈API集成到sender.py中。通过`bp_ltp_interface.py`作为中间层，将`dtn_ion.py`中的API函数进行了封装，以便在自适应优化框架中使用。

## 架构设计

### 模块结构

```
dtn_ion.py (BP/LTP原始API)
    ↓
bp_ltp_interface.py (BP/LTP接口封装)
    ↓
sender.py (发送节点A)
```

## API函数映射表

| dtn_ion.py函数 | bp_ltp_interface.py方法 | 功能 | 调用位置 |
|---|---|---|---|
| `configure_network()` | `configure_link_parameters()` | 配置链路参数（带宽、延时、丢包率） | `apply_protocol_params()` |
| `calculate_packet_loss()` | 内部调用 | 根据BER计算丢包率 | `configure_link_parameters()` |
| `calculate_ltp_sessions()` | `apply_protocol_parameters()` | 计算LTP会话数量 | `apply_protocol_parameters()` |
| `setup_ltp_span()` | `apply_protocol_parameters()` | 配置LTP span参数 | `apply_protocol_parameters()` |
| `setup_contact()` | `setup_transmission_contact()` | 设置transmission contact | `transmit_data()` |
| `send_bpdriver_command()` | `transmit_data_via_bp_ltp()` | 执行BP driver传输命令 | `transmit_data()` |
| `extract_time_from_ionadmin()` | 内部调用 | 获取节点启动时间 | `setup_transmission_contact()` |

## 核心工作流程

### 1. 链路参数配置流程

```
获取链路状态 (CSV配置文件)
    ↓
request_optimized_params()
    ↓
apply_protocol_params()
    ↓
configure_link_parameters()
    ├─ 计算丢包率 (calculate_packet_loss)
    └─ 配置网络 (configure_network via tc命令)
```

### 2. 协议参数应用流程

```
apply_protocol_params() 接收优化参数
    ↓
apply_protocol_parameters()
    ├─ 计算LTP会话数 (calculate_ltp_sessions)
    ├─ 配置LTP span (setup_ltp_span)
    └─ 返回成功/失败状态
```

### 3. 数据传输流程

```
transmit_data()
    ↓
[BP/LTP启用?]
    ├─ 是 → setup_transmission_contact()
    │        ↓
    │     transmit_data_via_bp_ltp()
    │        ├─ 计算bundle数量
    │        ├─ 执行send_bpdriver_command()
    │        └─ 返回发送时间戳
    └─ 否 → 使用TCP模拟模式
```

## BPLTPInterface类详解

### 初始化参数

```python
BPLTPInterface(
    source_eid="ipn:1.1",           # 源节点EID
    destination_eid=2,               # 目标节点EID（数字）
    dest_addr="192.168.1.2",        # 目标IP地址
    dest_udp_addr="192.168.1.2:1113", # 目标UDP地址
    bp_ttl=3600                     # Bundle生存时间（秒）
)
```

### 主要方法

#### 1. `configure_link_parameters()`

**作用**：根据链路状态配置网络参数

**参数**：
- `bit_error_rate`: 误码率
- `delay_ms`: 延时（毫秒）
- `transmission_rate_mbps`: 传输速率（Mbps）
- `data_size`: 数据大小（用于计算丢包率）

**调用的dtn_ion函数**：
```python
calculate_packet_loss()  # 计算丢包率
configure_network()      # 配置tc命令
```

#### 2. `apply_protocol_parameters()`

**作用**：应用协议栈参数到LTP

**参数**：
- `bundle_size`: Bundle大小
- `ltp_block_size`: LTP Block大小
- `ltp_segment_size`: LTP Segment大小
- `session_count`: LTP会话数（可选，为None时自动计算）
- `data_size`: 数据大小
- `delay_ms`: 延时
- `transmission_rate_mbps`: 传输速率

**调用的dtn_ion函数**：
```python
calculate_ltp_sessions()  # 计算会话数（如果未指定）
setup_ltp_span()          # 配置LTP span
```

#### 3. `setup_transmission_contact()`

**作用**：设置transmission contact

**参数**：
- `transmission_rate_mbps`: 传输速率

**调用的dtn_ion函数**：
```python
extract_time_from_ionadmin()  # 获取节点启动时间
setup_contact()               # 设置contact
```

#### 4. `transmit_data_via_bp_ltp()`

**作用**：通过BP/LTP协议栈发送数据

**参数**：
- `data_size`: 数据大小
- `transmission_rate_mbps`: 传输速率

**调用的dtn_ion函数**：
```python
send_bpdriver_command()  # 执行BP driver命令
```

## 发送节点A的集成

### 初始化

```python
sender = SenderNode(
    receiver_host='192.168.1.2',
    receiver_port=5001,
    optimizer_host='192.168.1.3',
    optimizer_port=5002,
    config_file='network_config.csv',
    source_eid='ipn:1.1',
    destination_eid=2,
    dest_udp_addr='192.168.1.2:1113',
    use_bp_ltp=True  # 启用BP/LTP
)
```

### 传输周期内的调用顺序

```
1. generate_business_request()  # 从CSV获取数据大小
2. get_link_state()            # 从CSV获取链路状态
3. request_optimized_params()  # 向优化器请求参数
4. apply_protocol_params()     # 应用参数
   ├─ configure_link_parameters()
   └─ apply_protocol_parameters()
5. transmit_data()             # 传输数据
   ├─ setup_transmission_contact()
   └─ transmit_data_via_bp_ltp()
6. send_metadata()             # 发送元数据到节点B
```

## 使用模式

### 模式1：BP/LTP实际部署

```bash
python3 sender.py \
  --use-bp-ltp \
  --source-eid "ipn:1.1" \
  --destination-eid 2 \
  --dest-udp-addr "192.168.1.2:1113" \
  --receiver-host 192.168.1.2 \
  --interval 10
```

**特点**：
- 使用真实的BP/LTP协议栈
- 调用ION系统命令
- 实际的网络配置（tc命令）
- 真实的数据传输

### 模式2：模拟模式（默认）

```bash
python3 sender.py \
  --simulate \
  --receiver-host 192.168.1.2 \
  --interval 10
```

**特点**：
- 不调用BP/LTP API
- 使用TCP模拟数据传输
- 用于测试和调试
- 不依赖ION系统

### 模式3：混合模式

```bash
python3 sender.py \
  --use-bp-ltp \
  --receiver-host 192.168.1.2 \
  --interval 10
```

**特点**：
- 尝试使用BP/LTP
- 如果BP/LTP初始化失败或缺少dtn_ion模块，自动回退到模拟模式
- 最灵活的部署方案

## 错误处理

系统采用分层错误处理策略：

### 层1：接口层（bp_ltp_interface.py）

```python
try:
    setup_ltp_span(...)
except Exception as e:
    print(f"[错误] 配置LTP span失败: {e}")
    return False
```

### 层2：节点层（sender.py）

```python
try:
    bp_send_time = self.bp_ltp_interface.transmit_data_via_bp_ltp(...)
    if bp_send_time > 0:
        # 成功
        return start_timestamp, True
except Exception as e:
    print(f"[警告] BP/LTP传输异常: {e}，回退到模拟模式")
    # 回退到TCP模拟
```

### 层3：系统层

```python
if self.use_bp_ltp and self.bp_ltp_interface:
    try:
        # 使用BP/LTP
    except:
        self.use_bp_ltp = False  # 禁用BP/LTP
else:
    # 使用模拟模式
```

## 参数流向示例

### 示例传输周期

假设从CSV文件读取如下配置：
```
data_size: 20480 bytes
bit_error_rate: 1e-4
delay_ms: 150.0
transmission_rate_mbps: 8.0
```

**流向过程**：

1. **获取参数**
   ```
   CSV → [20480, 1e-4, 150, 8.0]
   ```

2. **请求优化**
   ```
   优化器 → [bundle_size: 2048, ltp_block_size: 1024,
             ltp_segment_size: 512, session_count: 4]
   ```

3. **配置链路**
   ```
   configure_link_parameters():
   - 计算丢包率: calculate_packet_loss(1e-4, 512)
   - 配置网络: configure_network(192.168.1.2, 8000000, 150, X%)
   ```

4. **配置协议**
   ```
   apply_protocol_parameters():
   - 计算会话: calculate_ltp_sessions(150, 2048, 20480, 1024, 1000000)
   - 配置span: setup_ltp_span(2, sessions, 512, 1024, 5, 192.168.1.2:1113)
   ```

5. **设置contact**
   ```
   setup_transmission_contact():
   - 获取时间: extract_time_from_ionadmin(1, 2)
   - 设置contact: setup_contact(time, 1, 2, 1000000)
   ```

6. **发送数据**
   ```
   transmit_data_via_bp_ltp():
   - 计算cycles: ceil(20480/2048) = 10
   - 执行命令: send_bpdriver_command(10, "ipn:1.1", "ipn:2.1",
                                    2048, 3600, 1000000)
   ```

## 故障排除

### 问题1：无法找到bp_ltp_interface模块

**原因**：bp_ltp_interface.py不在Python路径中

**解决**：
```bash
# 确保bp_ltp_interface.py与sender.py在同一目录
ls -la /root/agent/bp_ltp_interface.py

# 或使用完整路径
PYTHONPATH=/root/agent python3 sender.py --use-bp-ltp
```

### 问题2：BP/LTP初始化失败

**原因**：dtn_ion.py导入失败或ION系统未启动

**解决**：
```bash
# 检查dtn_ion.py
python3 -c "from dtn_ion import configure_network; print('OK')"

# 检查ION系统
ionstop
ionstart
```

### 问题3：网络配置命令失败

**原因**：需要sudo权限

**解决**：
```bash
# 授予必要的权限
sudo visudo
# 添加: your_user ALL=(ALL) NOPASSWD: /sbin/tc

# 或以sudo运行
sudo python3 sender.py --use-bp-ltp
```

### 问题4：传输失败但无错误信息

**原因**：可能是接收端未启动

**解决**：
```bash
# 确保接收端已启动
python3 receiver.py --listen-port 5001

# 查看日志
tail -f sender.log
```

## 性能优化建议

1. **参数调优**
   - 根据延时调整会话数
   - 根据误码率调整segment大小
   - 根据数据大小调整bundle大小

2. **网络配置**
   - 使用hierarchical token bucket (HTB) 进行带宽限制
   - 使用netem进行延时和丢包模拟

3. **监控**
   - 监控LTP会话状态
   - 记录传输时间
   - 分析优化器的学习曲线

## 总结

通过这个集成框架，sender.py能够：
1. ✅ 从CSV获取多种网络场景
2. ✅ 向优化器请求自适应参数
3. ✅ 使用BP/LTP API配置协议栈
4. ✅ 真实传输数据或模拟传输
5. ✅ 支持灵活的部署模式
6. ✅ 提供完善的错误处理

这使得整个系统可以在真实BP/LTP环境和模拟环境中无缝运行。