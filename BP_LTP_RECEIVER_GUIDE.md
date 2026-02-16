# BP/LTP接收器集成指南

## 概述

接收节点B已集成BP/LTP接收功能，使用`bp_ltp_receiver_interface.py`作为中间层，封装了`dtn_ion.py`中的`run_bpcounter_and_monitor()`函数。

## 架构设计

### 模块结构

```
dtn_ion.py (BP/LTP接收API)
    ↓
bp_ltp_receiver_interface.py (BP/LTP接收器接口封装)
    ↓
receiver.py (接收节点B)
```

## 核心概念

### 1. Bundle数量计算

发送端发送数据时的计算：
```
nbr_of_cycles = ceil(data_size / bundle_size)
```

接收端接收数据时的计算：
```
max_count = ceil(data_size / bundle_size)
```

两者应该相等，以确保完整接收。

### 2. 接收流程

```
发送端发起传输
    ↓
接收端接收元数据（数据大小、bundle大小）
    ↓
接收端计算bundle数量
    ↓
接收端启动bpcounter监听线程
    ↓
发送端通过BP/LTP发送数据
    ↓
bpcounter监听并统计接收的bundle
    ↓
接收端获取接收完成时间
    ↓
计算业务交付时间
    ↓
生成训练记录
```

## BPLTPReceiverInterface类详解

### 初始化参数

```python
BPLTPReceiverInterface(
    own_eid_number=2  # 本节点EID数字，对应 ipn:2.1
)
```

### 主要方法

#### 1. `calculate_bundle_count()`

**作用**：根据数据大小和bundle大小计算bundle数量

**参数**：
- `data_size`: 总数据大小（bytes）
- `bundle_size`: 单个bundle大小（bytes）

**返回**：
- bundle数量（即bpcounter的max_count参数）

**公式**：
```
bundle_count = ceil(data_size / bundle_size)
```

**调用**：
```python
bundle_count = bp_ltp_receiver.calculate_bundle_count(20480, 1024)
# 结果: 20
```

#### 2. `monitor_reception()`

**作用**：启动bpcounter监听BP/LTP数据接收过程

**参数**：
- `max_count`: 要接收的bundle数量

**返回**：
- `(bpcounter_report, stop_time)`: bpcounter输出报告和停止时间戳

**调用的dtn_ion函数**：
```python
run_bpcounter_and_monitor(source, max_count)
```

**功能**：
- 执行bpcounter命令
- 持续监听接收过程
- 返回完整报告和停止时间戳

#### 3. `parse_bpcounter_report()`

**作用**：解析bpcounter报告提取性能指标

**参数**：
- `report`: bpcounter的输出报告

**返回**：
- 包含性能指标的字典

**指标**：
- `total_bundles_received`: 接收的总bundle数
- `total_bytes_received`: 接收的总字节数
- `delivery_rate`: 传输成功率

## ReceiverNode集成

### 初始化

```python
receiver = ReceiverNode(
    listen_port=5001,
    optimizer_host='192.168.1.3',
    optimizer_port=5003,
    own_eid_number=2,
    use_bp_ltp=True  # 启用BP/LTP
)
```

### 新增属性

```python
self.reception_thread        # BP/LTP接收线程
self.reception_event         # 接收完成事件
self.reception_result        # 接收结果存储
self.bp_ltp_receiver         # BP/LTP接收器接口实例
```

### 新增方法

#### 1. `start_bp_ltp_reception(data_size, bundle_size)`

**作用**：启动BP/LTP接收监听

**调用位置**：`handle_metadata()`

**流程**：
```
计算bundle数量
    ↓
创建接收线程
    ↓
启动bpcounter监听
```

#### 2. `_bp_ltp_reception_thread(bundle_count)`

**作用**：BP/LTP接收监听线程

**功能**：
```
启动bpcounter监听
    ↓
等待接收完成
    ↓
解析接收报告
    ↓
保存接收结果
    ↓
设置完成事件
```

### 工作流程

#### 步骤1：接收数据传输头

```python
def handle_data_transmission(data):
    start_timestamp = data.get("start_timestamp")
    data_size = data.get("data_size")
    # 记录开始时间和数据大小
```

#### 步骤2：接收元数据并启动监听

```python
def handle_metadata(data):
    # 获取元数据
    data_size = data.get("data_size")
    bundle_size = protocol_params.get("bundle_size")

    # 启动BP/LTP接收
    if self.use_bp_ltp:
        self.start_bp_ltp_reception(data_size, bundle_size)

    # 等待接收完成
    if self.reception_event.wait(timeout=300):
        end_timestamp = self.reception_result.get("stop_time")
```

#### 步骤3：计算交付时间

```python
delivery_time_ms = (end_timestamp - start_timestamp) * 1000
```

#### 步骤4：生成训练记录

```python
self.logger.record_transmission(
    data_size=data_size,
    bit_error_rate=link_state.get("bit_error_rate"),
    delay_ms=link_state.get("delay_ms"),
    transmission_rate_mbps=link_state.get("transmission_rate_mbps"),
    bundle_size=bundle_size,
    ltp_block_size=protocol_params.get("ltp_block_size"),
    ltp_segment_size=protocol_params.get("ltp_segment_size"),
    session_count=protocol_params.get("session_count"),
    delivery_time_ms=delivery_time_ms
)
```

## 使用模式

### 模式1：BP/LTP实际部署

```bash
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001
```

**特点**：
- 使用真实的bpcounter监听
- 记录实际接收时间
- 完整的性能指标

### 模式2：模拟模式（默认）

```bash
python3 receiver.py \
  --simulate \
  --listen-port 5001
```

**特点**：
- 使用TCP接收
- 使用当前时间作为完成时间
- 不依赖ION系统

### 模式3：混合模式（推荐）

```bash
python3 receiver.py \
  --use-bp-ltp \
  --own-eid-number 2 \
  --listen-port 5001
```

**特点**：
- 优先尝试使用BP/LTP
- 失败时自动回退到模拟模式

## 完整的端到端流程

### 时间线

```
T0: 发送端获取业务请求和链路状态
T0+ε: 发送端向优化器请求参数
T0+2ε: 发送端应用优化参数和链路配置
T0+3ε: 发送端通过BP/LTP开始发送（发送时间戳 T0+3ε）
    ↓
T0+3ε: 接收端收到元数据（包含开始时间戳 T0+3ε）
T0+3ε: 接收端启动bpcounter监听线程
    ↓
T0+3ε ~ T1: BP/LTP数据传输过程
    ↓
T1: bpcounter监听结束（完成时间戳 T1）
T1+δ: 接收端计算交付时间 = T1 - (T0+3ε)
T1+δ: 接收端生成训练记录
T1+δ: 发送记录到优化器
```

## 参数流向示例

### 示例：接收端处理过程

假设接收到以下元数据：
```json
{
  "data_size": 20480,
  "protocol_params": {
    "bundle_size": 1024,
    "ltp_block_size": 512,
    "ltp_segment_size": 256,
    "session_count": 4
  },
  "link_state": {
    "bit_error_rate": 1e-4,
    "delay_ms": 150.0,
    "transmission_rate_mbps": 8.0
  }
}
```

**流程**：

1. **计算bundle数量**
   ```
   bundle_count = ceil(20480 / 1024) = 20
   ```

2. **启动bpcounter监听**
   ```
   run_bpcounter_and_monitor("ipn:2.1", 20)
   ```

3. **等待接收完成**
   ```
   bpcounter监听过程中：
   - 检测到接收的bundle逐个到达
   - 当接收到第20个bundle时完成
   - 记录完成时间戳 T1
   ```

4. **计算交付时间**
   ```
   start_timestamp = T0 + 3ε
   end_timestamp = T1
   delivery_time_ms = (T1 - (T0+3ε)) * 1000
   ```

5. **生成训练记录**
   ```
   {
     "input": {
       "data_size": 20480,
       "bit_error_rate": 1e-4,
       "delay_ms": 150.0,
       "transmission_rate_mbps": 8.0
     },
     "output": {
       "bundle_size": 1024,
       "ltp_block_size": 512,
       "ltp_segment_size": 256,
       "session_count": 4
     },
     "performance": {
       "delivery_time_ms": <计算值>
     }
   }
   ```

## 线程管理

### 接收线程生命周期

```
主线程: handle_metadata()
    ↓
创建: _bp_ltp_reception_thread()
    ↓
后台运行: 执行bpcounter监听
    ↓
设置事件: 接收完成时设置reception_event
    ↓
主线程检测: reception_event.wait()返回
    ↓
继续执行: 计算交付时间和生成记录
```

### 超时处理

```python
# 最多等待300秒
if self.reception_event.wait(timeout=300):
    # 正常完成
    end_timestamp = self.reception_result.get("stop_time")
else:
    # 超时，使用当前时间
    print("[警告] BP/LTP接收超时")
    end_timestamp = time.time()
```

## 错误处理

### 场景1：BP/LTP接口初始化失败

```python
try:
    self.bp_ltp_receiver = BPLTPReceiverInterface(...)
except Exception as e:
    print(f"[警告] 初始化BP/LTP接收器接口失败: {e}")
    self.use_bp_ltp = False
    # 回退到模拟模式
```

### 场景2：bpcounter监听失败

```python
try:
    report, stop_time = run_bpcounter_and_monitor(...)
except Exception as e:
    print(f"[错误] 监听接收失败: {e}")
    self.reception_result["success"] = False
    # 继续执行，不中断主流程
```

### 场景3：接收超时

```python
if not self.reception_event.wait(timeout=300):
    print(f"[警告] BP/LTP接收超时")
    end_timestamp = time.time()
    # 使用当前时间作为完成时间
```

## 性能优化建议

1. **监听线程优化**
   - 使用后台守护线程（daemon=True）
   - 设置合理的超时时间
   - 及时清理线程资源

2. **时间同步**
   - 确保发送端和接收端时间同步
   - 使用NTP进行时间校准
   - 考虑网络延时

3. **接收监听**
   - 合理设置bpcounter的max_count
   - 监听报告的详细程度平衡
   - 性能指标的收集和处理

## 故障排除

### 问题1：bpcounter命令找不到

**原因**：ION系统未安装或不在PATH中

**解决**：
```bash
# 检查bpcounter是否存在
which bpcounter

# 或手动运行
sudo bpcounter ipn:2.1 20
```

### 问题2：接收超时

**原因**：
- 发送端未发送数据
- 网络连接问题
- bpcounter进程卡住

**解决**：
- 检查发送端日志
- 检查网络连接
- 手动kill bpcounter进程

### 问题3：时间戳不准确

**原因**：
- 发送端和接收端时间不同步
- 处理延时过长

**解决**：
- 同步系统时间：`ntpdate -s time.server`
- 优化处理流程

### 问题4：记录数据丢失

**原因**：
- 优化器未运行
- 网络连接中断
- 缓冲区溢出

**解决**：
- 确保优化器运行
- 检查网络连接
- 增加缓冲区大小

## 总结

接收器集成实现了：
1. ✅ 从元数据计算bundle数量
2. ✅ 启动bpcounter监听线程
3. ✅ 等待接收完成并获取完成时间
4. ✅ 计算业务交付时间
5. ✅ 生成完整的训练记录
6. ✅ 支持BP/LTP和模拟两种模式
7. ✅ 完善的错误处理和超时机制

这使得接收端能够准确记录传输性能，为强化学习优化器提供真实的训练数据。
