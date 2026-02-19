#!/usr/bin/env python3
"""
DQN离线训练模块
从CSV数据集加载历史数据进行模型预训练
"""

import csv
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 添加路径以导入DQN优化器
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mode_dqn_v2_gpu import DQNOptimizerGPU, TORCH_AVAILABLE, DEVICE
    GPU_MODE = True
except ImportError:
    print("[警告] 无法导入GPU版本，尝试导入CPU版本")
    try:
        from mode_dqn_v2 import DQNOptimizer
        GPU_MODE = False
    except ImportError:
        print("[错误] 无法导入DQN优化器")
        sys.exit(1)


class OfflineTrainer:
    """离线训练器"""

    def __init__(self, model_save_path: str = None, use_gpu: bool = True):
        """
        初始化离线训练器

        Args:
            model_save_path: 模型保存路径
            use_gpu: 是否使用GPU（如果可用）
        """
        self.use_gpu = use_gpu and GPU_MODE and TORCH_AVAILABLE
        self.model_save_path = model_save_path or "/root/agent/computer/dqn_model.pth"

        # 初始化优化器
        if self.use_gpu:
            print(f"[初始化] 使用GPU模式进行离线训练")
            self.optimizer = DQNOptimizerGPU()
        else:
            print(f"[初始化] 使用CPU模式进行离线训练")
            self.optimizer = DQNOptimizer()

        print(f"[配置] 模型保存路径: {self.model_save_path}")

    def load_dataset_from_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        从CSV文件加载训练数据集

        CSV格式要求:
            - data_size: 数据量大小（bytes）
            - bit_error_rate: 误码率
            - delay_ms: 延时（毫秒）
            - transmission_rate_mbps: 传输速率（Mbps）
            - bundle_size: Bundle大小（bytes）
            - ltp_block_size: LTP Block大小（bytes）
            - ltp_segment_size: LTP Segment大小（bytes）
            - session_count: 会话数量
            - delivery_time_ms: 业务交付时间（毫秒）

        Args:
            csv_file: CSV文件路径

        Returns:
            训练记录列表
        """
        records = []

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row_idx, row in enumerate(reader):
                    try:
                        # 解析input字段
                        input_data = {
                            "data_size": int(float(row['data_size'])),
                            "bit_error_rate": float(row['bit_error_rate']),
                            "delay_ms": float(row['delay_ms']),
                            "transmission_rate_mbps": float(row['transmission_rate_mbps'])
                        }

                        # 解析output字段
                        output_data = {
                            "bundle_size": int(float(row['bundle_size'])),
                            "ltp_block_size": int(float(row['ltp_block_size'])),
                            "ltp_segment_size": int(float(row['ltp_segment_size'])),
                            "session_count": int(float(row['session_count']))
                        }

                        # 解析performance字段
                        performance_data = {
                            "delivery_time_ms": float(row['delivery_time_ms'])
                        }

                        # 构造训练记录
                        record = {
                            "input": input_data,
                            "output": output_data,
                            "performance": performance_data,
                            "timestamp": float(row.get('timestamp', time.time()))
                        }

                        records.append(record)

                    except (KeyError, ValueError) as e:
                        print(f"[警告] 第{row_idx + 2}行数据解析失败: {e}")
                        continue

            print(f"[数据加载] 成功从 {csv_file} 加载 {len(records)} 条训练记录")
            return records

        except FileNotFoundError:
            print(f"[错误] CSV文件不存在: {csv_file}")
            return []
        except Exception as e:
            print(f"[错误] 加载CSV文件失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def validate_dataset(self, records: List[Dict[str, Any]]) -> bool:
        """
        验证数据集的有效性

        Args:
            records: 训练记录列表

        Returns:
            数据集是否有效
        """
        if not records:
            print("[验证] ❌ 数据集为空")
            return False

        print(f"[验证] 数据集包含 {len(records)} 条记录")

        # 检查必需字段
        required_fields = ['input', 'output', 'performance']
        for idx, record in enumerate(records[:5]):  # 只检查前5条
            for field in required_fields:
                if field not in record:
                    print(f"[验证] ❌ 记录{idx}缺少字段: {field}")
                    return False

        # 统计数据分布
        delivery_times = [r['performance']['delivery_time_ms'] for r in records]
        bundle_sizes = [r['output']['bundle_size'] for r in records]
        block_sizes = [r['output']['ltp_block_size'] for r in records]

        print(f"[验证] 交付时间范围: {min(delivery_times):.2f} ~ {max(delivery_times):.2f} ms")
        print(f"[验证] Bundle大小范围: {min(bundle_sizes)} ~ {max(bundle_sizes)} bytes")
        print(f"[验证] Block大小范围: {min(block_sizes)} ~ {max(block_sizes)} bytes")

        # 检查约束条件
        constraint_violations = 0
        for idx, record in enumerate(records):
            bundle = record['output']['bundle_size']
            block = record['output']['ltp_block_size']

            if block < bundle:
                print(f"[验证] ⚠️  记录{idx}: block({block}) < bundle({bundle})")
                constraint_violations += 1
            elif block % bundle != 0:
                print(f"[验证] ⚠️  记录{idx}: block({block}) % bundle({bundle}) != 0")
                constraint_violations += 1

        if constraint_violations > 0:
            print(f"[验证] ⚠️  发现 {constraint_violations} 条违反约束的记录")
            print(f"[验证] 这些记录可能会影响训练效果")

        print(f"[验证] ✅ 数据集验证通过")
        return True

    def train_from_dataset(self, records: List[Dict[str, Any]],
                          epochs: int = 1,
                          batch_size: int = 50,
                          save_interval: int = 0) -> bool:
        """
        使用数据集训练模型

        Args:
            records: 训练记录列表
            epochs: 训练轮数（整个数据集遍历次数）
            batch_size: 批次大小（每批处理多少条记录）
            save_interval: 保存间隔（每N批保存一次，0表示不定期保存）

        Returns:
            训练是否成功
        """
        if not records:
            print("[训练] ❌ 数据集为空，无法训练")
            return False

        print(f"\n{'='*70}")
        print(f"开始离线训练")
        print(f"{'='*70}")
        print(f"训练配置:")
        print(f"  • 数据集大小: {len(records)} 条记录")
        print(f"  • 训练轮数: {epochs}")
        print(f"  • 批次大小: {batch_size}")
        print(f"  • 每轮批次数: {len(records) // batch_size + (1 if len(records) % batch_size else 0)}")
        print(f"  • 模式: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"{'='*70}\n")

        total_batches = 0
        start_time = time.time()

        try:
            for epoch in range(epochs):
                print(f"\n[Epoch {epoch + 1}/{epochs}] 开始训练")

                # 打乱数据集（每轮训练前）
                import random
                shuffled_records = records.copy()
                random.shuffle(shuffled_records)

                # 分批训练
                for batch_idx in range(0, len(shuffled_records), batch_size):
                    batch = shuffled_records[batch_idx:batch_idx + batch_size]
                    batch_num = batch_idx // batch_size + 1
                    total_batches += 1

                    print(f"\n[Epoch {epoch + 1}/{epochs}, 批次 {batch_num}] 训练 {len(batch)} 条记录")

                    # 调用优化器的批量训练方法
                    self.optimizer.batch_update_model(batch)

                    # 定期保存（如果设置了保存间隔）
                    if save_interval > 0 and total_batches % save_interval == 0:
                        self.save_model(f"{self.model_save_path}.epoch{epoch+1}.batch{batch_num}")
                        print(f"[保存] 已保存中间模型")

                print(f"\n[Epoch {epoch + 1}/{epochs}] 完成")
                print(f"  • 当前探索率: {self.optimizer.epsilon:.4f}")
                print(f"  • 模型版本: {self.optimizer.model_version}")
                if hasattr(self.optimizer, 'episode_rewards') and self.optimizer.episode_rewards:
                    avg_reward = np.mean(list(self.optimizer.episode_rewards))
                    print(f"  • 平均奖励: {avg_reward:.4f}")

            # 训练完成
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"离线训练完成")
            print(f"{'='*70}")
            print(f"训练统计:")
            print(f"  • 总训练批次: {total_batches}")
            print(f"  • 总耗时: {elapsed:.2f} 秒")
            print(f"  • 平均每批: {elapsed / total_batches:.2f} 秒")
            print(f"  • 最终模型版本: {self.optimizer.model_version}")
            print(f"  • 最终探索率: {self.optimizer.epsilon:.4f}")
            if hasattr(self.optimizer, 'episode_rewards') and self.optimizer.episode_rewards:
                final_avg_reward = np.mean(list(self.optimizer.episode_rewards))
                print(f"  • 最终平均奖励: {final_avg_reward:.4f}")
            print(f"{'='*70}\n")

            return True

        except Exception as e:
            print(f"\n[错误] 训练过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_model(self, save_path: str = None):
        """
        保存训练好的模型

        Args:
            save_path: 保存路径（如果为None，使用默认路径）
        """
        save_path = save_path or self.model_save_path

        try:
            if self.use_gpu:
                # GPU模式：保存PyTorch模型
                checkpoint = {
                    'policy_net_state_dict': self.optimizer.policy_net.state_dict(),
                    'target_net_state_dict': self.optimizer.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
                    'model_version': self.optimizer.model_version,
                    'training_steps': self.optimizer.training_steps,
                    'epsilon': self.optimizer.epsilon,
                    'valid_action_pairs': self.optimizer.valid_action_pairs,
                    'episode_rewards': list(self.optimizer.episode_rewards)
                }
                torch.save(checkpoint, save_path)
                print(f"[保存] ✅ GPU模型已保存到: {save_path}")

            else:
                # CPU模式：保存NumPy权重
                import pickle
                checkpoint = {
                    'network_weights': {
                        'W1': self.optimizer.network.W1,
                        'b1': self.optimizer.network.b1,
                        'W2': self.optimizer.network.W2,
                        'b2': self.optimizer.network.b2,
                        'W3': self.optimizer.network.W3,
                        'b3': self.optimizer.network.b3,
                    },
                    'target_weights': {
                        'W1': self.optimizer.network.target_W1,
                        'b1': self.optimizer.network.target_b1,
                        'W2': self.optimizer.network.target_W2,
                        'b2': self.optimizer.network.target_b2,
                        'W3': self.optimizer.network.target_W3,
                        'b3': self.optimizer.network.target_b3,
                    },
                    'model_version': self.optimizer.model_version,
                    'training_steps': self.optimizer.training_steps,
                    'epsilon': self.optimizer.epsilon,
                    'valid_action_pairs': self.optimizer.valid_action_pairs,
                    'episode_rewards': list(self.optimizer.episode_rewards)
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"[保存] ✅ CPU模型已保存到: {save_path}")

        except Exception as e:
            print(f"[错误] 保存模型失败: {e}")
            import traceback
            traceback.print_exc()

    def load_model(self, load_path: str = None):
        """
        加载已训练的模型

        Args:
            load_path: 模型文件路径（如果为None，使用默认路径）
        """
        load_path = load_path or self.model_save_path

        try:
            if self.use_gpu:
                # GPU模式：加载PyTorch模型
                checkpoint = torch.load(load_path, map_location=DEVICE)

                self.optimizer.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.optimizer.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.optimizer.model_version = checkpoint.get('model_version', 0)
                self.optimizer.training_steps = checkpoint.get('training_steps', 0)
                self.optimizer.epsilon = checkpoint.get('epsilon', 0.1)

                if 'episode_rewards' in checkpoint:
                    from collections import deque
                    self.optimizer.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)

                # 更新推理快照
                self.optimizer._create_inference_snapshot()

                print(f"[加载] ✅ GPU模型已从 {load_path} 加载")
                print(f"  • 模型版本: {self.optimizer.model_version}")
                print(f"  • 训练步数: {self.optimizer.training_steps}")
                print(f"  • 探索率: {self.optimizer.epsilon:.4f}")

            else:
                # CPU模式：加载NumPy权重
                import pickle
                with open(load_path, 'rb') as f:
                    checkpoint = pickle.load(f)

                # 恢复网络权重
                self.optimizer.network.W1 = checkpoint['network_weights']['W1']
                self.optimizer.network.b1 = checkpoint['network_weights']['b1']
                self.optimizer.network.W2 = checkpoint['network_weights']['W2']
                self.optimizer.network.b2 = checkpoint['network_weights']['b2']
                self.optimizer.network.W3 = checkpoint['network_weights']['W3']
                self.optimizer.network.b3 = checkpoint['network_weights']['b3']

                # 恢复目标网络权重
                self.optimizer.network.target_W1 = checkpoint['target_weights']['W1']
                self.optimizer.network.target_b1 = checkpoint['target_weights']['b1']
                self.optimizer.network.target_W2 = checkpoint['target_weights']['W2']
                self.optimizer.network.target_b2 = checkpoint['target_weights']['b2']
                self.optimizer.network.target_W3 = checkpoint['target_weights']['W3']
                self.optimizer.network.target_b3 = checkpoint['target_weights']['b3']

                self.optimizer.model_version = checkpoint.get('model_version', 0)
                self.optimizer.training_steps = checkpoint.get('training_steps', 0)
                self.optimizer.epsilon = checkpoint.get('epsilon', 0.1)

                if 'episode_rewards' in checkpoint:
                    from collections import deque
                    self.optimizer.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)

                print(f"[加载] ✅ CPU模型已从 {load_path} 加载")
                print(f"  • 模型版本: {self.optimizer.model_version}")
                print(f"  • 训练步数: {self.optimizer.training_steps}")
                print(f"  • 探索率: {self.optimizer.epsilon:.4f}")

        except FileNotFoundError:
            print(f"[错误] 模型文件不存在: {load_path}")
        except Exception as e:
            print(f"[错误] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # ==================== 配置参数（在此修改） ====================
    dataset_path = '/root/agent/training_data.csv'              # 训练数据集CSV文件路径
    epochs = 10                                                 # 训练轮数
    batch_size = 50                                             # 批次大小
    save_path = '/root/agent/computer/dqn_model_pretrained.pth'  # 模型保存路径
    save_interval = 0                                           # 保存间隔（每N批保存一次，0表示只在结束时保存）
    load_model = None                                           # 加载已有模型继续训练（None表示从头训练）
    use_gpu = True                                              # 是否使用GPU（如果可用）

    # 如果要加载已有模型继续训练，取消下面一行的注释并修改路径：
    # load_model = '/root/agent/computer/dqn_model_stage1.pth'
    # ============================================================

    print("="*60)
    print("DQN离线训练工具")
    print("="*60)
    print(f"数据集路径: {dataset_path}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"模型保存路径: {save_path}")
    print(f"保存间隔: {save_interval if save_interval > 0 else '仅在结束时保存'}")
    print(f"预加载模型: {load_model if load_model else '不加载（从头训练）'}")
    print(f"使用GPU: {'是' if use_gpu else '否（仅CPU）'}")
    print("="*60)
    print()

    # 创建离线训练器
    trainer = OfflineTrainer(
        model_save_path=save_path,
        use_gpu=use_gpu
    )

    # 如果指定了加载模型，先加载
    if load_model:
        print(f"[预加载] 加载已有模型: {load_model}")
        trainer.load_model(load_model)
        print()

    # 加载数据集
    print(f"[数据集] 加载训练数据: {dataset_path}")
    records = trainer.load_dataset_from_csv(dataset_path)

    if not records:
        print("[错误] 无法加载数据集，退出")
        sys.exit(1)

    # 验证数据集
    if not trainer.validate_dataset(records):
        print("[错误] 数据集验证失败，退出")
        sys.exit(1)

    # 开始训练
    success = trainer.train_from_dataset(
        records=records,
        epochs=epochs,
        batch_size=batch_size,
        save_interval=save_interval
    )

    if success:
        # 保存最终模型
        trainer.save_model()
        print(f"\n✅ 训练完成！模型已保存到: {save_path}")
        print(f"\n使用方法：")
        print(f"  1. 在mode_dqn_v2_gpu.py的main()函数中设置pretrained_model路径")
        print(f"  2. 启动优化器即可自动加载预训练模型")
        print(f"     示例: pretrained_model = '{save_path}'")
    else:
        print(f"\n❌ 训练失败")
        sys.exit(1)


if __name__ == "__main__":
    main()