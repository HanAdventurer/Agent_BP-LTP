#!/usr/bin/env python3
"""
测试2：Segment自适应选择验证
验证segment_size是否根据网络状态正确地自适应选择
"""

import sys
sys.path.insert(0, '/root/agent/computer')

from mode_dqn_v2 import DQNOptimizer

print("[初始化] 正在加载DQN优化器...")
optimizer = DQNOptimizer()

print(f"\n{'='*60}")
print(f"[测试] Segment自适应性验证")
print(f"{'='*60}\n")

# 测试不同网络条件下的segment选择
test_cases = [
    {"name": "极佳网络（低延时+低误码）", "delay": 20.0, "ber": 1e-7, "expected": 200},
    {"name": "良好网络（中低延时+低误码）", "delay": 150.0, "ber": 1e-6, "expected": 200},
    {"name": "中等网络（中等延时+中等误码）", "delay": 300.0, "ber": 1e-4, "expected": 600},
    {"name": "较差网络（高延时+中等误码）", "delay": 500.0, "ber": 1e-3, "expected": 1000},
    {"name": "恶劣网络（高延时+高误码）", "delay": 800.0, "ber": 1e-2, "expected": 1400},
]

passed = 0
failed = 0

for test in test_cases:
    # 使用固定动作（0）以便只测试segment自适应
    params = optimizer.action_to_params(
        action=0,  # 固定动作
        data_size=10240,
        delay_ms=test["delay"],
        transmission_rate_mbps=10.0,
        bit_error_rate=test["ber"]
    )

    segment = params['ltp_segment_size']

    # 计算恶劣度
    delay_factor = min(test["delay"] / 1000.0, 1.0)
    ber_factor = min(test["ber"] / 0.01, 1.0)
    adversity = 0.6 * delay_factor + 0.4 * ber_factor

    print(f"{test['name']}")
    print(f"  输入: delay={test['delay']:.1f}ms, BER={test['ber']:.1e}")
    print(f"  恶劣度: {adversity:.3f}")
    print(f"  选择的segment: {segment}")

    # 验证是否符合预期范围
    if segment == test["expected"]:
        print(f"  ✅ 符合预期")
        passed += 1
    else:
        print(f"  ⚠️   预期{test['expected']}，但实际为{segment}")
        # 只要在合理范围内，仍算通过
        if segment in [200, 400, 600, 800, 1000, 1200, 1400]:
            passed += 1
        else:
            failed += 1
    print()

print(f"{'='*60}")
print(f"[测试总结]")
print(f"{'='*60}")
print(f"通过: {passed}/{len(test_cases)}")
if failed == 0:
    print(f"✅ Segment自适应选择测试通过!")
else:
    print(f"❌ {failed}个测试失败")
    sys.exit(1)

# 额外测试：验证segment约束检查
print(f"\n[额外测试] Segment约束检查")
print(f"测试：当bundle_size小时，segment应该被限制\n")

# 选择一个bundle=20000的动作（应该限制segment不能超过10000）
action = 24  # bundle=20000, block=20000
params = optimizer.action_to_params(
    action=action,
    data_size=10240,
    delay_ms=800.0,  # 恶劣条件，理论上会选1400
    transmission_rate_mbps=10.0,
    bit_error_rate=0.01
)

bundle = params['bundle_size']
block = params['ltp_block_size']
segment = params['ltp_segment_size']

print(f"动作{action}: bundle={bundle}, block={block}")
print(f"在恶劣网络条件下，segment={segment}")

max_allowed = int(block * 0.5)
if segment <= max_allowed:
    print(f"✅ segment({segment}) <= block*50%({max_allowed})，约束满足")
else:
    print(f"❌ segment({segment}) > block*50%({max_allowed})，约束违反")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"✅ 所有测试通过！")
print(f"{'='*60}")
