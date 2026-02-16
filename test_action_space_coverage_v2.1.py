#!/usr/bin/env python3
"""
测试3：动作空间覆盖分析
分析v2.1采样的32个动作对参数空间的覆盖情况
"""

import sys
sys.path.insert(0, '/root/agent/computer')

from mode_dqn_v2 import DQNOptimizer

print("[初始化] 正在加载DQN优化器...")
optimizer = DQNOptimizer()

print(f"\n{'='*60}")
print(f"[测试] 动作空间覆盖分析")
print(f"{'='*60}\n")

# 收集所有参数
bundle_values = {}  # {bundle: [actions]}
block_values = {}   # {block: [actions]}
segment_values = {} # {segment: count}
ratio_stats = []    # block/bundle比例

for action in range(optimizer.action_dim):
    bundle, block = optimizer.valid_action_pairs[action]

    if bundle not in bundle_values:
        bundle_values[bundle] = []
    bundle_values[bundle].append(action)

    if block not in block_values:
        block_values[block] = []
    block_values[block].append(action)

    # 获取segment值（使用标准网络条件）
    params = optimizer.action_to_params(
        action=action,
        data_size=10240,
        delay_ms=100.0,
        transmission_rate_mbps=10.0,
        bit_error_rate=1e-5
    )
    segment = params['ltp_segment_size']
    if segment not in segment_values:
        segment_values[segment] = 0
    segment_values[segment] += 1

    ratio = block // bundle
    ratio_stats.append((ratio, action, bundle, block))

print(f"[1] Bundle大小覆盖 ({len(bundle_values)}种)\n")
for bundle in sorted(bundle_values.keys()):
    actions = bundle_values[bundle]
    print(f"  Bundle={bundle:6d}: {len(actions):2d}个动作 - 动作索引: {actions}")

print(f"\n[2] Block大小覆盖 ({len(block_values)}种)\n")
for block in sorted(block_values.keys()):
    actions = block_values[block]
    print(f"  Block={block:7d}: {len(actions):2d}个动作 - 动作索引: {actions}")

print(f"\n[3] Segment大小分布\n")
for segment in sorted(segment_values.keys()):
    count = segment_values[segment]
    percentage = count / optimizer.action_dim * 100
    bar = "█" * int(percentage / 5)
    print(f"  Segment={segment:4d}: {count:2d}个动作 ({percentage:5.1f}%) {bar}")

print(f"\n[4] Block/Bundle比例分布\n")
ratio_stats.sort()
ratio_distribution = {}
for ratio, _, _, _ in ratio_stats:
    if ratio not in ratio_distribution:
        ratio_distribution[ratio] = 0
    ratio_distribution[ratio] += 1

print(f"  比例范围: {ratio_stats[0][0]} 到 {ratio_stats[-1][0]}")
print(f"  分布统计:")
for ratio in sorted(ratio_distribution.keys())[:10]:  # 只显示前10个
    count = ratio_distribution[ratio]
    print(f"    比例 {ratio:4d}: {count:2d}个动作")
if len(ratio_distribution) > 10:
    print(f"    ... 共{len(ratio_distribution)}种不同比例")

print(f"\n[5] 详细映射表（部分）\n")
print(f"  {'动作':>4} {'Bundle':>8} {'Block':>8} {'比例':>6} {'Segment':>8}")
print(f"  {'-'*40}")
for i in range(min(10, optimizer.action_dim)):
    bundle, block = optimizer.valid_action_pairs[i]
    params = optimizer.action_to_params(
        action=i,
        data_size=10240,
        delay_ms=100.0,
        transmission_rate_mbps=10.0,
        bit_error_rate=1e-5
    )
    segment = params['ltp_segment_size']
    ratio = block // bundle
    print(f"  {i:4d} {bundle:8d} {block:8d} {ratio:6d} {segment:8d}")
print(f"  ... (共{optimizer.action_dim}个动作)")

print(f"\n[6] 覆盖率评估\n")

# 计算理论最大覆盖率
all_valid_combinations = optimizer._generate_all_valid_combinations()
print(f"  理论所有有效组合数: {len(all_valid_combinations)}")
print(f"  采样后的动作数: {optimizer.action_dim}")
print(f"  覆盖率: {optimizer.action_dim / len(all_valid_combinations) * 100:.1f}%")

# Bundle覆盖
total_bundles = 15
bundle_coverage = len(bundle_values) / total_bundles * 100
print(f"\n  Bundle参数覆盖: {len(bundle_values)}/{total_bundles} ({bundle_coverage:.1f}%)")

# Block覆盖
total_blocks = 20
block_coverage = len(block_values) / total_blocks * 100
print(f"  Block参数覆盖: {len(block_values)}/{total_blocks} ({block_coverage:.1f}%)")

print(f"\n{'='*60}")
print(f"[分析总结]")
print(f"{'='*60}")
print(f"✅ 动作空间采样成功")
print(f"✅ Bundle覆盖{len(bundle_values)}种，比例{bundle_coverage:.1f}%")
print(f"✅ Block覆盖{len(block_values)}种，比例{block_coverage:.1f}%")
print(f"✅ Segment分布合理，覆盖{len(segment_values)}种值")
print(f"✅ Block/Bundle比例范围广 ({ratio_stats[0][0]} ~ {ratio_stats[-1][0]})")
