#!/usr/bin/env python3
"""
优化器/算力电脑C (Optimizer Computer C) - DQN v2.2
功能：
1. 接收节点A的优化请求
2. 基于DQN强化学习模型生成优化参数（Bundle、Block、Segment）
3. 自动计算LTP会话数量（不是训练得出）
4. 将优化参数发送给节点A
5. 接收节点B的训练记录
6. 使用训练记录进行DQN模型的训练和更新

关键改进（v2.2）：
- 动作空间包含(bundle, block, segment)三元组
- bundle_size支持15种选项：1k-100k
- block_size支持20种选项：20k-1000k
- segment_size作为第四个独立维度：7种选项（200-1400）
- 约束保证：block >= bundle AND block % bundle == 0 AND segment <= block*50%
- session_count通过calculate_ltp_sessions()计算，不参与训练
"""

import socket
import json
import time
import struct
import threading
import numpy as np
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque
import random


def calculate_ltp_sessions(delay: float, bundle_size: int, file_size: int,
                           block_size: int, trans_rate: float) -> int:
    """
    计算LTP会话数量（来自dtn_ion.py）

    这是一个确定性的计算函数，不需要训练学习。
    基于网络延时、数据特性和协议参数计算最优会话数。

    Args:
        delay: 延时（毫秒）
        bundle_size: Bundle大小（bytes）
        file_size: 文件大小（bytes）
        block_size: LTP Block大小（bytes）
        trans_rate: 传输速率（bytes/s）

    Returns:
        最优LTP会话数量
    """
    total_bundles = math.ceil(file_size / bundle_size)
    bundles_per_block = math.ceil(block_size / bundle_size)
    ltp_blocks = math.ceil(total_bundles / bundles_per_block)
    times = delay/500 + ((block_size + 20) / trans_rate)
    ltp_sessions = math.ceil((times * trans_rate) / (block_size + 20)) + 1
    ltp_sessions = min(ltp_sessions, ltp_blocks + 1, 20)
    return ltp_sessions


class ExperienceReplay:
    """经验回放缓冲区 - DQN核心组件"""

    def __init__(self, max_size: int = 10000):
        """
        初始化经验回放缓冲区

        Args:
            max_size: 最大存储容量
        """
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: Tuple):
        """
        添加经验
        experience: (state, action, reward, next_state, done)
        """
        self.memory.append(experience)

    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """
        随机采样一批经验

        Args:
            batch_size: 批大小

        Returns:
            经验列表
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def is_full(self) -> bool:
        """检查缓冲区是否满"""
        return len(self.memory) >= self.max_size


class DQNNetwork:
    """DQN神经网络 - 使用简化的全连接网络"""

    def __init__(self, state_dim: int = 4, action_dim: int = 9, hidden_dim: int = 128):
        """
        初始化DQN网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度（从16改为9：3×3）
            hidden_dim: 隐层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 主网络权重
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b3 = np.zeros((1, action_dim))

        # 目标网络权重（用于稳定训练）
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()

        self.learning_rate = 0.001

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)

    def forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        前向传播

        Args:
            state: 状态
            use_target: 是否使用目标网络

        Returns:
            Q值
        """
        if use_target:
            W1, b1 = self.target_W1, self.target_b1
            W2, b2 = self.target_W2, self.target_b2
            W3, b3 = self.target_W3, self.target_b3
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2
            W3, b3 = self.W3, self.b3

        # 确保state是2D数组
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # 第一层
        self.z1 = np.dot(state, W1) + b1
        self.a1 = self.relu(self.z1)

        # 第二层
        self.z2 = np.dot(self.a1, W2) + b2
        self.a2 = self.relu(self.z2)

        # 输出层
        q_values = np.dot(self.a2, W3) + b3

        return q_values

    def backward(self, state: np.ndarray, q_targets: np.ndarray):
        """
        反向传播

        Args:
            state: 状态
            q_targets: 目标Q值
        """
        batch_size = state.shape[0]

        # 前向传播
        q_pred = self.forward(state, use_target=False)

        # 计算损失
        loss = np.mean((q_pred - q_targets) ** 2)

        # 反向传播
        dq = 2 * (q_pred - q_targets) / batch_size

        dW3 = np.dot(self.a2.T, dq)
        db3 = np.sum(dq, axis=0, keepdims=True)

        da2 = np.dot(dq, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)

        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = np.dot(state.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 更新权重
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        return loss

    def update_target_network(self, tau: float = 0.001):
        """
        使用软更新策略更新目标网络

        Args:
            tau: 软更新系数（0.001表示目标网络更新缓慢）
        """
        self.target_W1 = tau * self.W1 + (1 - tau) * self.target_W1
        self.target_b1 = tau * self.b1 + (1 - tau) * self.target_b1
        self.target_W2 = tau * self.W2 + (1 - tau) * self.target_W2
        self.target_b2 = tau * self.b2 + (1 - tau) * self.target_b2
        self.target_W3 = tau * self.W3 + (1 - tau) * self.target_W3
        self.target_b3 = tau * self.b3 + (1 - tau) * self.target_b3


class RewardCalculator:
    """奖励函数计算器 - 基于传输性能的多维度奖励"""

    def __init__(self,
                 max_delivery_time_ms: float = 5000.0,
                 min_delivery_time_ms: float = 100.0):
        """
        初始化奖励计算器

        Args:
            max_delivery_time_ms: 最大业务交付时间（用于归一化）
            min_delivery_time_ms: 最小业务交付时间（用于归一化）
        """
        self.max_delivery_time = max_delivery_time_ms
        self.min_delivery_time = min_delivery_time_ms

    def calculate_reward(self,
                        delivery_time_ms: float,
                        data_size: int,
                        bit_error_rate: float,
                        delay_ms: float) -> float:
        """
        计算总体奖励

        Args:
            delivery_time_ms: 业务交付时间（毫秒）
            data_size: 数据大小（bytes）
            bit_error_rate: 误码率
            delay_ms: 链路延时（毫秒）

        Returns:
            奖励值（归一化到-1~1之间）
        """
        # 1. 交付时间奖励（权重0.5）：最小化交付时间
        time_reward = self._calculate_time_reward(delivery_time_ms)

        # 2. 吞吐量奖励（权重0.3）：最大化传输吞吐量
        throughput_reward = self._calculate_throughput_reward(
            data_size, delivery_time_ms
        )

        # 3. 鲁棒性奖励（权重0.2）：在恶劣条件下的表现
        robustness_reward = self._calculate_robustness_reward(
            bit_error_rate, delay_ms, delivery_time_ms
        )

        # 加权组合
        total_reward = (
            0.5 * time_reward +
            0.3 * throughput_reward +
            0.2 * robustness_reward
        )

        return total_reward

    def _calculate_time_reward(self, delivery_time_ms: float) -> float:
        """计算交付时间奖励"""
        if delivery_time_ms <= self.min_delivery_time:
            return 1.0
        elif delivery_time_ms >= self.max_delivery_time:
            return -1.0
        else:
            time_ratio = (delivery_time_ms - self.min_delivery_time) / \
                         (self.max_delivery_time - self.min_delivery_time)
            return 1.0 - 2.0 * time_ratio

    def _calculate_throughput_reward(self, data_size: int, delivery_time_ms: float) -> float:
        """计算吞吐量奖励"""
        throughput_mbps = (data_size * 8) / (delivery_time_ms / 1000) / 1_000_000
        max_expected_throughput = 100.0
        throughput_normalized = min(throughput_mbps / max_expected_throughput, 1.0)
        return 2.0 * throughput_normalized - 1.0

    def _calculate_robustness_reward(self,
                                     bit_error_rate: float,
                                     delay_ms: float,
                                     delivery_time_ms: float) -> float:
        """计算鲁棒性奖励"""
        ber_factor = min(bit_error_rate * 1e6, 1.0)
        delay_factor = min(delay_ms / 500.0, 1.0)
        adversity = (ber_factor + delay_factor) / 2.0

        if adversity > 0.5:
            if delivery_time_ms < self.max_delivery_time * 0.7:
                return 0.5
            elif delivery_time_ms < self.max_delivery_time:
                return 0.0
            else:
                return -0.5
        else:
            if delivery_time_ms < self.min_delivery_time * 2:
                return 0.3
            else:
                return 0.0


class DQNOptimizer:
    """DQN优化器 - 基于深度Q网络的协议参数优化（v2：session_count通过公式计算）"""

    def __init__(self):
        """初始化DQN优化器（v2.2：segment作为第四个独立维度）"""
        self.state_dim = 4  # data_size, BER, delay, rate

        # v2.2改进：生成包含segment的三元组动作空间
        all_valid_tuples = self._generate_all_valid_3tuples()
        self.valid_action_tuples = all_valid_tuples  # 直接使用所有有效组合
        self.action_dim = len(self.valid_action_tuples)

        # 初始化网络（action_dim现在是动态的）
        self.network = DQNNetwork(self.state_dim, self.action_dim)

        # 经验回放
        self.experience_replay = ExperienceReplay(max_size=10000)

        # 奖励计算器
        self.reward_calculator = RewardCalculator()

        # 训练参数
        self.epsilon = 0.1  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # 折扣因子
        self.batch_size = 32
        self.update_frequency = 100

        # 统计信息
        self.model_version = 0
        self.training_steps = 0
        self.episode_rewards = deque(maxlen=100)

        print("[DQN优化器v2.2] 初始化完成（segment作为第四维度）")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim} (所有满足约束的三元组)")
        print(f"  网络隐层维度: 128")
        print(f"  session_count: 通过calculate_ltp_sessions()计算")

        # 打印动作空间覆盖统计
        bundle_values = set()
        block_values = set()
        segment_values = set()
        for bundle, block, segment in self.valid_action_tuples:
            bundle_values.add(bundle)
            block_values.add(block)
            segment_values.add(segment)

        print(f"  Bundle大小覆盖: {len(bundle_values)}种（{sorted(bundle_values)[:3]}...）")
        print(f"  Block大小覆盖: {len(block_values)}种（{sorted(block_values)[:3]}...）")
        print(f"  Segment大小覆盖: {len(segment_values)}种（{sorted(segment_values)}）")

    def _generate_all_valid_3tuples(self) -> List[Tuple[int, int, int]]:
        """
        生成所有满足约束的(bundle_size, block_size, segment_size)三元组
        约束:
        1. block_size >= bundle_size
        2. block_size % bundle_size == 0
        3. segment_size <= block_size * 50%
        """
        bundle_sizes = [
            1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000,
            20000, 24000, 30000, 40000, 60000, 80000, 100000
        ]
        block_sizes = [
            20000, 40000, 60000, 80000, 100000, 120000, 140000,
            160000, 180000, 200000, 220000, 240000, 260000, 280000,
            300000, 350000, 400000, 450000, 500000, 1000000
        ]
        segment_sizes = [200, 400, 600, 800, 1000, 1200, 1400]

        valid_tuples = []
        for bundle in bundle_sizes:
            for block in block_sizes:
                # 约束1和2：block >= bundle AND block % bundle == 0
                if block >= bundle and block % bundle == 0:
                    for segment in segment_sizes:
                        valid_tuples.append((bundle, block, segment))

        return valid_tuples

    def discretize_state(self, state: Dict[str, float]) -> np.ndarray:
        """
        将连续状态转换为归一化向量

        Args:
            state: 包含data_size、bit_error_rate、delay_ms、transmission_rate_mbps的字典

        Returns:
            归一化的状态向量
        """
        data_size_normalized = min(state.get("data_size", 0) / 100000, 1.0)
        ber_normalized = min(state.get("bit_error_rate", 0) * 1e6, 1.0)
        delay_normalized = min(state.get("delay_ms", 0) / 500, 1.0)
        rate_normalized = min(state.get("transmission_rate_mbps", 0) / 100, 1.0)

        state_vector = np.array([
            data_size_normalized,
            ber_normalized,
            delay_normalized,
            rate_normalized
        ], dtype=np.float32)

        return state_vector

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        使用ε-贪心策略选择动作

        Args:
            state: 归一化的状态向量
            training: 是否处于训练模式

        Returns:
            动作索引 (0-8)
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            action = np.random.randint(0, self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.network.forward(state)
            action = np.argmax(q_values[0])

        return action

    def action_to_params(self, action: int,
                        data_size: int,
                        delay_ms: float,
                        transmission_rate_mbps: float) -> Dict[str, int]:
        """
        将动作索引转换为协议参数（v2.2：segment作为独立维度）

        关键改进：
        1. 从三元组中直接获取bundle、block、segment
        2. session_count通过calculate_ltp_sessions()计算

        Args:
            action: 动作索引 (0 to action_dim-1)
            data_size: 数据大小（用于计算session）
            delay_ms: 延时（用于计算session）
            transmission_rate_mbps: 传输速率（用于计算session）

        Returns:
            协议参数字典
        """
        # 约束检查
        if action >= self.action_dim:
            raise ValueError(f"动作{action}超出范围[0, {self.action_dim-1}]")

        # 从三元组中直接查表获取参数
        bundle_size, ltp_block_size, ltp_segment_size = self.valid_action_tuples[action]

        # 计算LTP会话数量（不是训练得出！）
        trans_rate_bytes = transmission_rate_mbps * 1_000_000 / 8  # 转换为bytes/s
        ltp_sessions = calculate_ltp_sessions(
            delay=delay_ms,
            bundle_size=bundle_size,
            file_size=data_size,
            block_size=ltp_block_size,
            trans_rate=trans_rate_bytes
        )

        params = {
            "bundle_size": bundle_size,
            "ltp_block_size": ltp_block_size,
            "ltp_segment_size": ltp_segment_size,
            "session_count": ltp_sessions  # 通过公式计算
        }

        return params

    def optimize_params(self, request_data: Dict[str, Any]) -> Dict[str, int]:
        """
        生成优化参数

        Args:
            request_data: 包含data_size和link_state的请求

        Returns:
            优化的协议参数
        """
        # 提取必要信息
        data_size = request_data.get("data_size", 10240)
        link_state = request_data.get("link_state", {})

        # 构造状态
        state = {
            "data_size": data_size,
            "bit_error_rate": link_state.get("bit_error_rate", 1e-5),
            "delay_ms": link_state.get("delay_ms", 100.0),
            "transmission_rate_mbps": link_state.get("transmission_rate_mbps", 10.0)
        }

        # 转换状态
        state_vector = self.discretize_state(state)

        # 选择动作
        action = self.select_action(state_vector, training=True)

        # 转换为参数（segment已包含在三元组中）
        params = self.action_to_params(
            action=action,
            data_size=data_size,
            delay_ms=state["delay_ms"],
            transmission_rate_mbps=state["transmission_rate_mbps"]
        )

        # 日志输出
        print(f"[DQN优化] 动作={action}/{self.action_dim}, Bundle={params['bundle_size']}, "
              f"Block={params['ltp_block_size']}, Segment={params['ltp_segment_size']}, "
              f"Session={params['session_count']}(计算)")

        return params

    def store_experience(self,
                        state: Dict[str, float],
                        action: int,
                        reward: float,
                        next_state: Dict[str, float],
                        done: bool):
        """
        存储经验到回放缓冲区

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否完成
        """
        state_vector = self.discretize_state(state)
        next_state_vector = self.discretize_state(next_state)

        self.experience_replay.add((state_vector, action, reward, next_state_vector, done))

    def train_batch(self):
        """使用经验回放缓冲区中的一批经验进行训练"""
        if len(self.experience_replay.memory) < self.batch_size:
            return 0.0

        # 采样批次
        batch = self.experience_replay.sample_batch(self.batch_size)

        # 解包
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # 计算目标Q值
        q_targets = self.network.forward(states)
        q_next = self.network.forward(next_states, use_target=True)

        for i in range(len(batch)):
            if dones[i]:
                q_targets[i, actions[i]] = rewards[i]
            else:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        # 反向传播
        loss = self.network.backward(states, q_targets)

        # 更新目标网络
        self.network.update_target_network(tau=0.001)

        self.training_steps += 1

        return loss

    def batch_update_model(self, records: List[Dict[str, Any]]):
        """
        使用一批训练记录更新模型

        Args:
            records: 包含input/output/performance的记录列表
        """
        print(f"\n[DQN训练v2] 开始使用 {len(records)} 条记录进行批量训练")

        for i, record in enumerate(records):
            try:
                # 解包记录
                input_data = record.get("input", {})
                output_data = record.get("output", {})
                performance = record.get("performance", {})

                state = {
                    "data_size": input_data.get("data_size", 0),
                    "bit_error_rate": input_data.get("bit_error_rate", 0),
                    "delay_ms": input_data.get("delay_ms", 0),
                    "transmission_rate_mbps": input_data.get("transmission_rate_mbps", 0)
                }

                delivery_time_ms = performance.get("delivery_time_ms", 5000.0)

                # 计算奖励
                reward = self.reward_calculator.calculate_reward(
                    delivery_time_ms=delivery_time_ms,
                    data_size=input_data.get("data_size", 0),
                    bit_error_rate=input_data.get("bit_error_rate", 0),
                    delay_ms=input_data.get("delay_ms", 0)
                )

                # 查找对应的动作（仅根据bundle和block）
                action = self._find_action_from_params(output_data)

                # 存储经验
                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,
                    done=False
                )

                # 进行训练
                loss = self.train_batch()

                if i % 10 == 0:
                    print(f"  [记录{i+1}/{len(records)}] 奖励: {reward:.4f}, "
                          f"交付时间: {delivery_time_ms:.2f}ms, Loss: {loss:.6f}")

                self.episode_rewards.append(reward)

            except Exception as e:
                print(f"  [错误] 处理记录 {i} 失败: {e}")

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 增加模型版本
        self.model_version += 1

        # 输出统计信息
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        print(f"[DQN训练完成v2] 模型版本: {self.model_version}, "
              f"平均奖励: {avg_reward:.4f}, "
              f"探索率ε: {self.epsilon:.4f}, "
              f"训练步数: {self.training_steps}")

    def _find_action_from_params(self, params: Dict[str, int]) -> int:
        """
        从参数字典反向查找动作索引
        v2.2改进：支持三元组匹配

        Args:
            params: 协议参数字典

        Returns:
            匹配的动作索引
        """
        bundle = params.get("bundle_size", 1024)
        block = params.get("ltp_block_size", 512)
        segment = params.get("ltp_segment_size", 200)

        try:
            # 第一优先级：完全匹配 (bundle, block, segment)
            for idx, (b, bl, s) in enumerate(self.valid_action_tuples):
                if b == bundle and bl == block and s == segment:
                    return idx

            # 第二优先级：匹配bundle和block（segment可能不同）
            for idx, (b, bl, s) in enumerate(self.valid_action_tuples):
                if b == bundle and bl == block:
                    return idx

            # 第三优先级：随机选择一个有效动作
            return np.random.randint(0, self.action_dim)

        except Exception as e:
            print(f"[警告] 查找动作索引失败: {e}，使用随机动作")
            return np.random.randint(0, self.action_dim)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_version": self.model_version,
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.experience_replay.memory),
            "avg_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0
        }


class OptimizerServer:
    """优化器服务器 - 接收请求和训练数据"""

    def __init__(self,
                 param_request_port: int = 5002,
                 record_receive_port: int = 5003):
        """
        初始化优化器服务器

        Args:
            param_request_port: 接收参数请求的端口
            record_receive_port: 接收训练记录的端口
        """
        self.param_request_port = param_request_port
        self.record_receive_port = record_receive_port

        self.dqn_optimizer = DQNOptimizer()
        self.running = True

    def handle_param_request(self, request_data: Dict[str, Any]) -> Dict[str, int]:
        """
        处理参数优化请求

        Args:
            request_data: 来自节点A的请求数据

        Returns:
            优化后的参数
        """
        try:
            link_state = request_data.get("link_state", {})

            print(f"[参数请求] 数据量: {request_data.get('data_size')} bytes, "
                  f"误码率: {link_state.get('bit_error_rate')}, "
                  f"延时: {link_state.get('delay_ms')}ms, "
                  f"速率: {link_state.get('transmission_rate_mbps')}Mbps")

            # 生成优化参数（包含计算的session_count）
            optimized_params = self.dqn_optimizer.optimize_params(request_data)

            return optimized_params

        except Exception as e:
            print(f"[错误] 处理参数请求失败: {e}")
            return {}

    def param_request_server(self):
        """处理参数请求的服务器线程"""
        print(f"[参数请求服务器] 启动 (监听端口 {self.param_request_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.param_request_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

                    # 接收请求
                    length_data = client_socket.recv(4)
                    if not length_data:
                        continue

                    message_length = struct.unpack('!I', length_data)[0]
                    request_data = b''
                    while len(request_data) < message_length:
                        chunk = client_socket.recv(min(4096, message_length - len(request_data)))
                        if not chunk:
                            break
                        request_data += chunk

                    # 解析请求
                    request = json.loads(request_data.decode('utf-8'))
                    print(f"\n[新请求] 来自 {client_address}, 模型版本: {self.dqn_optimizer.model_version}")

                    # 处理请求
                    optimized_params = self.handle_param_request(request)

                    # 发送响应
                    response = {
                        "status": "success",
                        "optimized_params": optimized_params,
                        "model_version": self.dqn_optimizer.model_version,
                        "model_info": self.dqn_optimizer.get_model_info(),
                        "timestamp": time.time()
                    }

                    response_json = json.dumps(response)
                    response_message = response_json.encode('utf-8')

                    client_socket.sendall(struct.pack('!I', len(response_message)))
                    client_socket.sendall(response_message)

                    print(f"[参数响应] 已发送优化参数")

                    client_socket.close()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[错误] 处理参数请求失败: {e}")

        finally:
            server_socket.close()

    def record_receive_server(self):
        """处理训练记录接收的服务器线程"""
        print(f"[记录接收服务器] 启动 (监听端口 {self.record_receive_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.record_receive_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

                    # 接收消息
                    length_data = client_socket.recv(4)
                    if not length_data:
                        continue

                    message_length = struct.unpack('!I', length_data)[0]
                    message_data = b''
                    while len(message_data) < message_length:
                        chunk = client_socket.recv(min(4096, message_length - len(message_data)))
                        if not chunk:
                            break
                        message_data += chunk

                    # 解析消息
                    message = json.loads(message_data.decode('utf-8'))
                    message_type = message.get("type")

                    if message_type == "training_records":
                        records = message.get("records", [])
                        print(f"\n[收到训练记录] {len(records)} 条来自 {client_address}")

                        # 使用记录更新模型
                        self.dqn_optimizer.batch_update_model(records)

                        # 发送确认
                        ack_message = "training_records_received"

                    else:
                        print(f"[警告] 未知的消息类型: {message_type}")
                        ack_message = "unknown_type"

                    client_socket.sendall(ack_message.encode('utf-8'))
                    client_socket.close()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[错误] 处理训练记录失败: {e}")

        finally:
            server_socket.close()

    def run(self):
        """运行优化器服务器"""
        print("="*60)
        print("DQN优化器服务器v2启动")
        print(f"参数请求服务器: 端口 {self.param_request_port}")
        print(f"记录接收服务器: 端口 {self.record_receive_port}")
        print("关键改进: session_count通过calculate_ltp_sessions()计算")
        print("="*60)

        # 启动两个服务器线程
        param_thread = threading.Thread(target=self.param_request_server, daemon=False)
        record_thread = threading.Thread(target=self.record_receive_server, daemon=False)

        param_thread.start()
        record_thread.start()

        try:
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n优化器停止")
            self.running = False

        # 等待线程完成
        param_thread.join(timeout=5)
        record_thread.join(timeout=5)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BP/LTP DQN优化器v2')
    parser.add_argument('--param-port', type=int, default=5002,
                        help='接收参数请求的端口')
    parser.add_argument('--record-port', type=int, default=5003,
                        help='接收训练记录的端口')

    args = parser.parse_args()

    optimizer = OptimizerServer(
        param_request_port=args.param_port,
        record_receive_port=args.record_port
    )

    optimizer.run()


if __name__ == "__main__":
    main()