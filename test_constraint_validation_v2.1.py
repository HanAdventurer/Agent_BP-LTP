#!/usr/bin/env python3
"""
测试1：DQN v2.1约束验证测试
验证所有采样的动作是否满足约束条件
"""

import sys
sys.path.insert(0, '/root/agent/computer')

try:
    from mode_dqn_v2 import DQNOptimizer
    print("[初始化] 正在加载DQN优化器...")
    optimizer = DQNOptimizer()
except Exception as e:
    print(f"[错误] 无法加载优化器: {e}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"[测试] 动作空间约束验证")
print(f"{'='*60}")
print(f"动作空间大小: {optimizer.action_dim}")
print(f"预期: 32个动作（采样得到）\n")

# 验证所有动作的约束
errors = []
warnings = []
valid_count = 0

for action in range(optimizer.action_dim):
    try:
        params = optimizer.action_to_params(
            action=action,
            data_size=10240,
            delay_ms=100.0,
            transmission_rate_mbps=10.0,
            bit_error_rate=1e-5
        )

        bundle = params['bundle_size']
        block = params['ltp_block_size']
        segment = params['ltp_segment_size']
        session = params['session_count']

        # 约束1: block >= bundle
        if block < bundle:
            errors.append(f"动作{action}: 约束违反 - block({block}) < bundle({bundle})")
            continue

        # 约束2: block % bundle == 0
        if block % bundle != 0:
            errors.append(f"动作{action}: 约束违反 - block({block}) % bundle({bundle}) != 0")
            continue

        # 约束3: segment在有效范围内
        if segment not in [200, 400, 600, 800, 1000, 1200, 1400]:
            errors.append(f"动作{action}: segment({segment})不在有效选项中")
            continue

        # 约束4: session > 0
        if session <= 0:
            errors.append(f"动作{action}: session({session}) <= 0")
            continue

        # 约束5: segment <= block * 50%
        max_segment = int(block * 0.5)
        if segment > max_segment:
            warnings.append(f"动作{action}: segment({segment}) > block*50%({max_segment})")

        valid_count += 1
        ratio = block // bundle
        print(f"✅ 动作{action:2d}: bundle={bundle:6d}, block={block:7d}, "
              f"ratio={ratio:3d}, segment={segment:4d}, session={session}")

    except Exception as e:
        errors.append(f"动作{action}: 异常 - {str(e)}")

# 总结
print(f"\n{'='*60}")
print(f"[测试总结]")
print(f"{'='*60}")

if errors:
    print(f"❌ 发现{len(errors)}个错误:")
    for err in errors:
        print(f"  - {err}")
    exit_code = 1
else:
    print(f"✅ 所有{optimizer.action_dim}个动作都通过约束验证!")
    exit_code = 0

if warnings:
    print(f"\n⚠️   发现{len(warnings)}个警告:")
    for warn in warnings:
        print(f"  - {warn}")

print(f"\n[覆盖统计]")
bundle_values = set()
block_values = set()
for action in range(optimizer.action_dim):
    bundle, block = optimizer.valid_action_pairs[action]
    bundle_values.add(bundle)
    block_values.add(block)

print(f"Bundle大小覆盖: {len(bundle_values)}种")
print(f"  {sorted(bundle_values)}")
print(f"Block大小覆盖: {len(block_values)}种")
print(f"  {sorted(block_values)}")

sys.exit(exit_code)
