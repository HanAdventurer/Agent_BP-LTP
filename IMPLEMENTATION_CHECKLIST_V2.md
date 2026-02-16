# DQN v2实现检查清单

## ✅ 快速验证清单

使用此清单验证v2系统是否正确实现。

---

## 📋 文件完整性检查

### 第1步：验证关键文件存在

- [ ] `/root/agent/computer/mode_dqn_v2.py` 存在
  ```bash
  ls -la /root/agent/computer/mode_dqn_v2.py
  # 应该显示文件大小 > 20KB
  ```

- [ ] `/root/agent/send/sender.py` 存在
  ```bash
  ls -la /root/agent/send/sender.py
  ```

- [ ] `/root/agent/receive/receiver.py` 存在
  ```bash
  ls -la /root/agent/receive/receiver.py
  ```

- [ ] `/root/agent/send/dtn_ion.py` 包含 `calculate_ltp_sessions()`
  ```bash
  grep -c "def calculate_ltp_sessions" /root/agent/send/dtn_ion.py
  # 应该输出：1
  ```

---

## 🔧 代码实现检查

### 第2步：验证mode_dqn_v2.py的核心组件

- [ ] **DQNNetwork类存在**
  ```bash
  grep -c "class DQNNetwork" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

- [ ] **ExperienceReplay类存在**
  ```bash
  grep -c "class ExperienceReplay" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

- [ ] **RewardCalculator类存在**
  ```bash
  grep -c "class RewardCalculator" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

- [ ] **DQNOptimizer类存在**
  ```bash
  grep -c "class DQNOptimizer" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

### 第3步：验证关键方法实现

- [ ] **action_to_params()方法包含calculate_ltp_sessions()调用**
  ```bash
  grep -A 20 "def action_to_params" /root/agent/computer/mode_dqn_v2.py | \
    grep -c "calculate_ltp_sessions"
  # 应该输出：1
  ```

- [ ] **状态离散化方法存在**
  ```bash
  grep -c "def discretize_state" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

- [ ] **动作选择方法存在**
  ```bash
  grep -c "def select_action" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

- [ ] **训练方法存在**
  ```bash
  grep -c "def train" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：1
  ```

---

## 🎯 动作空间检查

### 第4步：验证动作空间正确

- [ ] **动作空间大小为9**
  ```bash
  grep "self.action_dim = " /root/agent/computer/mode_dqn_v2.py
  # 应该显示：self.action_dim = 9
  ```

- [ ] **Bundle大小选项正确**
  ```bash
  grep -A 1 '"bundle_size":' /root/agent/computer/mode_dqn_v2.py | head -2
  # 应该显示：[1024, 2048, 4096]（3种选择）
  ```

- [ ] **Block大小选项正确**
  ```bash
  grep -A 1 '"ltp_block_size":' /root/agent/computer/mode_dqn_v2.py | head -2
  # 应该显示：[512, 1024, 2048]（3种选择）
  ```

- [ ] **Segment大小选项正确**
  ```bash
  grep -A 1 '"ltp_segment_size":' /root/agent/computer/mode_dqn_v2.py | head -2
  # 应该显示：[256, 512, 1024]（3种选择，跟随block）
  ```

---

## 📊 奖励函数检查

### 第5步：验证多维奖励函数

- [ ] **总奖励计算权重正确**
  ```bash
  grep "0.5 \* time_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该找到此行
  grep "0.3 \* throughput_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该找到此行
  grep "0.2 \* robustness_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该找到此行
  ```

- [ ] **时间奖励方法存在**
  ```bash
  grep -c "_calculate_time_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：2（定义和使用）
  ```

- [ ] **吞吐量奖励方法存在**
  ```bash
  grep -c "_calculate_throughput_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：2
  ```

- [ ] **鲁棒性奖励方法存在**
  ```bash
  grep -c "_calculate_robustness_reward" /root/agent/computer/mode_dqn_v2.py
  # 应该输出：2
  ```

---

## 🔗 接口兼容性检查

### 第6步：验证系统接口匹配

- [ ] **sender.py能够发送参数请求**
  ```bash
  grep -c "request_optimization_params" /root/agent/send/sender.py
  # 应该输出：1
  ```

- [ ] **receiver.py能够记录传输信息**
  ```bash
  grep -c "def record_transmission" /root/agent/receive/receiver.py
  # 应该输出：1
  ```

- [ ] **receiver.py能够发送训练记录**
  ```bash
  grep -c "def send_records" /root/agent/receive/receiver.py
  # 应该输出：1
  ```

- [ ] **optimizer能够监听参数请求**
  ```bash
  grep -c "param_server" /root/agent/computer/mode_dqn_v2.py
  # 应该输出 >= 1
  ```

- [ ] **optimizer能够监听训练记录**
  ```bash
  grep -c "record_server" /root/agent/computer/mode_dqn_v2.py
  # 应该输出 >= 1
  ```

---

## 🚀 运行时检查

### 第7步：验证系统能够启动

- [ ] **mode_dqn_v2.py能够导入所有依赖**
  ```bash
  python3 -c "import sys; sys.path.insert(0, '/root/agent/computer'); exec(open('/root/agent/computer/mode_dqn_v2.py').read())" 2>&1 | head -20
  # 应该看到初始化消息，不应该有导入错误
  ```

- [ ] **sender.py能够初始化**
  ```bash
  cd /root/agent/send && python3 sender.py --help 2>&1 | head -10
  # 应该显示帮助信息
  ```

- [ ] **receiver.py能够初始化**
  ```bash
  cd /root/agent/receive && python3 receiver.py --help 2>&1 | head -10
  # 应该显示帮助信息
  ```

---

## 🧪 功能测试

### 第8步：验证系统功能

#### 8.1 启动系统
```bash
# 终端1
cd /root/agent/receive && python3 receiver.py --simulate > /tmp/receiver.log 2>&1 &
RECEIVER_PID=$!

# 终端2
cd /root/agent/computer && python3 mode_dqn_v2.py > /tmp/optimizer.log 2>&1 &
OPTIMIZER_PID=$!

# 等待初始化
sleep 3

# 终端3
cd /root/agent/send && timeout 30 python3 sender.py --simulate --interval 5 > /tmp/sender.log 2>&1 &
SENDER_PID=$!

# 等待运行
sleep 35

# 清理进程
kill $SENDER_PID $OPTIMIZER_PID $RECEIVER_PID 2>/dev/null
```

#### 8.2 检查关键日志

- [ ] **参数请求被处理**
  ```bash
  grep -c "已发送优化参数" /tmp/optimizer.log
  # 应该输出 >= 1
  ```

- [ ] **session_count被计算**
  ```bash
  grep "已发送优化参数" /tmp/optimizer.log | head -1
  # 应该显示 session_count 值
  ```

- [ ] **训练记录被接收**
  ```bash
  grep -c "收到训练记录" /tmp/optimizer.log
  # 应该输出 >= 1
  ```

- [ ] **DQN训练执行**
  ```bash
  grep -c "DQN训练" /tmp/optimizer.log
  # 应该输出 >= 1
  ```

- [ ] **模型版本增加**
  ```bash
  grep "模型版本" /tmp/optimizer.log | tail -1
  # 应该显示 > 0
  ```

---

## 📈 性能验证

### 第9步：验证学习效果

```bash
# 运行10分钟
timeout 600 python3 /root/agent/send/sender.py --simulate --interval 5 > /tmp/sender.log 2>&1 &
SENDER_PID=$!

# 后台运行优化器和接收端
python3 /root/agent/computer/mode_dqn_v2.py > /tmp/optimizer.log 2>&1 &
OPTIMIZER_PID=$!

python3 /root/agent/receive/receiver.py --simulate > /tmp/receiver.log 2>&1 &
RECEIVER_PID=$!

# 等待完成
wait $SENDER_PID

# 清理
kill $OPTIMIZER_PID $RECEIVER_PID 2>/dev/null
```

- [ ] **平均奖励上升**
  ```bash
  FIRST=$(grep "平均奖励" /tmp/optimizer.log | head -1 | grep -oP "[-\d.]+$" | tail -1)
  LAST=$(grep "平均奖励" /tmp/optimizer.log | tail -1 | grep -oP "[-\d.]+$" | tail -1)

  echo "首次奖励: $FIRST"
  echo "最后奖励: $LAST"
  # 应该看到 LAST > FIRST（至少数值更小的负数或更大的正数）
  ```

- [ ] **探索率下降**
  ```bash
  FIRST=$(grep "探索率" /tmp/optimizer.log | head -1 | grep -oP "0\.\d+$" | tail -1)
  LAST=$(grep "探索率" /tmp/optimizer.log | tail -1 | grep -oP "0\.\d+$" | tail -1)

  echo "首次探索率: $FIRST"
  echo "最后探索率: $LAST"
  # 应该看到 LAST < FIRST
  ```

- [ ] **交付时间减少**
  ```bash
  grep "交付时间" /tmp/optimizer.log | head -5
  grep "交付时间" /tmp/optimizer.log | tail -5
  # 应该看到后期的时间普遍更小
  ```

- [ ] **模型版本增长**
  ```bash
  VERSIONS=$(grep "模型版本" /tmp/optimizer.log | grep -oP "模型版本: \K\d+" | sort -u | wc -l)
  echo "不同的模型版本: $VERSIONS"
  # 应该输出 >= 3
  ```

---

## 🔍 调试检查

### 第10步：如果出现问题

- [ ] **检查numpy是否安装**
  ```bash
  python3 -c "import numpy; print(numpy.__version__)"
  # 应该输出版本号
  ```

- [ ] **检查端口是否被占用**
  ```bash
  netstat -tlnp | grep -E "5001|5002|5003"
  # 如果有输出，说明端口已被占用
  ```

- [ ] **检查计算函数是否可调用**
  ```bash
  python3 << 'EOF'
  from root.agent.computer.mode_dqn_v2 import calculate_ltp_sessions
  result = calculate_ltp_sessions(100, 1024, 10240, 512, 1000000)
  print(f"Session count: {result}")
  EOF
  # 应该输出一个有效的数字
  ```

- [ ] **检查动作转换是否正确**
  ```bash
  python3 << 'EOF'
  from root.agent.computer.mode_dqn_v2 import DQNOptimizer
  opt = DQNOptimizer()

  for action in [0, 4, 8]:
      params = opt.action_to_params(action, 10240, 100, 10)
      print(f"Action {action}: {params}")
  EOF
  # 应该输出3个不同的参数组合，session_count都是有效的数字
  ```

---

## 📝 完成情况总结

### 检查表填写说明

1. **必须全部通过** (步骤1-9)
   - [ ] 步骤1：文件完整性
   - [ ] 步骤2：代码实现
   - [ ] 步骤3：关键方法
   - [ ] 步骤4：动作空间
   - [ ] 步骤5：奖励函数
   - [ ] 步骤6：接口兼容性
   - [ ] 步骤7：运行时检查
   - [ ] 步骤8：功能测试
   - [ ] 步骤9：性能验证

2. **如有任何项未通过**
   - 检查对应的第10步调试检查
   - 查阅相关文档
   - 参考迁移指南

3. **全部通过后**
   - ✅ v2系统已正确实现
   - ✅ 可以投入生产使用
   - ✅ 定期监控性能指标

---

## 🎯 快速验证脚本

将以下内容保存为 `verify_v2.sh`:

```bash
#!/bin/bash

echo "=========================================="
echo "DQN v2完整性验证脚本"
echo "=========================================="

PASS=0
FAIL=0

check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1 存在"
        ((PASS++))
    else
        echo "❌ $1 不存在"
        ((FAIL++))
    fi
}

check_grep() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo "✅ $1 包含 '$2'"
        ((PASS++))
    else
        echo "❌ $1 缺少 '$2'"
        ((FAIL++))
    fi
}

# 文件检查
echo -e "\n[文件检查]"
check_file "/root/agent/computer/mode_dqn_v2.py"
check_file "/root/agent/send/sender.py"
check_file "/root/agent/receive/receiver.py"

# 代码检查
echo -e "\n[代码检查]"
check_grep "/root/agent/computer/mode_dqn_v2.py" "class DQNNetwork"
check_grep "/root/agent/computer/mode_dqn_v2.py" "class ExperienceReplay"
check_grep "/root/agent/computer/mode_dqn_v2.py" "class DQNOptimizer"
check_grep "/root/agent/computer/mode_dqn_v2.py" "calculate_ltp_sessions"
check_grep "/root/agent/computer/mode_dqn_v2.py" "self.action_dim = 9"

# 摘要
echo -e "\n=========================================="
echo "验证结果: $PASS 项通过, $FAIL 项失败"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo "✅ v2系统完整性检查通过！"
    exit 0
else
    echo "❌ v2系统存在问题，请修复"
    exit 1
fi
```

使用方法:
```bash
chmod +x verify_v2.sh
./verify_v2.sh
```

---

**检查清单版本**: 1.0
**最后更新**: 2025年（当前）
**相关文档**: ARCHITECTURE_V2_SUMMARY.md, MIGRATION_V1_TO_V2.md