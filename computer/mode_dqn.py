#!/usr/bin/env python3
"""
优化器/算力电脑C (Optimizer Computer C) - DQN版本
功能：
1. 接收节点A的优化请求
2. 基于DQN强化学习模型生成优化参数
3. 将优化参数发送给节点A
4. 接收节点B的训练记录
5. 使用训练记录进行DQN模型的训练和更新
"""

import socket
import json
import time
import struct
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque
import random


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

    def __init__(self, state_dim: int = 4, action_dim: int = 16, hidden_dim: int = 128):
        """
        初始化DQN网络

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
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
        """
        计算交付时间奖励

        最小化交付时间：
        - 如果交付时间 < min_time，给予最大奖励1.0
        - 如果交付时间 > max_time，给予最小奖励-1.0
        - 线性插值中间值
        """
        if delivery_time_ms <= self.min_delivery_time:
            return 1.0
        elif delivery_time_ms >= self.max_delivery_time:
            return -1.0
        else:
            # 线性插值
            time_ratio = (delivery_time_ms - self.min_delivery_time) / \
                         (self.max_delivery_time - self.min_delivery_time)
            return 1.0 - 2.0 * time_ratio

    def _calculate_throughput_reward(self, data_size: int, delivery_time_ms: float) -> float:
        """
        计算吞吐量奖励

        吞吐量 = data_size / delivery_time
        - 高吞吐量给予正奖励
        - 低吞吐量给予负奖励
        """
        throughput_mbps = (data_size * 8) / (delivery_time_ms / 1000) / 1_000_000

        # 归一化（假设最大期望吞吐量100Mbps）
        max_expected_throughput = 100.0
        throughput_normalized = min(throughput_mbps / max_expected_throughput, 1.0)

        # 吞吐量奖励：高吞吐量为正，低吞吐量为负
        return 2.0 * throughput_normalized - 1.0

    def _calculate_robustness_reward(self,
                                     bit_error_rate: float,
                                     delay_ms: float,
                                     delivery_time_ms: float) -> float:
        """
        计算鲁棒性奖励

        在恶劣网络条件下保持良好性能：
        - 高误码率或高延时时，如果交付时间仍较短，给予额外奖励
        - 这鼓励算法在各种条件下都能工作良好
        """
        # 计算网络条件恶劣程度
        ber_factor = min(bit_error_rate * 1e6, 1.0)  # 归一化到0-1
        delay_factor = min(delay_ms / 500.0, 1.0)  # 归一化到0-1
        adversity = (ber_factor + delay_factor) / 2.0

        # 在恶劣条件下，交付时间相对较短给予更多奖励
        if adversity > 0.5:
            # 恶劣条件
            if delivery_time_ms < self.max_delivery_time * 0.7:
                return 0.5  # 在恶劣条件下表现良好，额外奖励
            elif delivery_time_ms < self.max_delivery_time:
                return 0.0
            else:
                return -0.5
        else:
            # 良好条件
            if delivery_time_ms < self.min_delivery_time * 2:
                return 0.3  # 在良好条件下也表现良好
            else:
                return 0.0


class DQNOptimizer:
    """DQN优化器 - 基于深度Q网络的协议参数优化"""

    def __init__(self):
        """初始化DQN优化器"""
        self.state_dim = 4  # data_size, BER, delay, rate
        self.action_dim = 16  # 4x4的动作空间

        # 动作映射
        self.action_space = {
            "bundle_size": [512, 1024, 2048, 4096],
            "ltp_block_size": [256, 512, 1024, 2048],
            "ltp_segment_size": [128, 256, 512, 1024],
            "session_count": [1, 2, 4, 8]
        }

        # 初始化网络
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

        print("[DQN优化器] 初始化完成")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  网络隐层维度: 128")

    def discretize_state(self, state: Dict[str, float]) -> np.ndarray:
        """
        将连续状态转换为归一化向量

        Args:
            state: 包含data_size、bit_error_rate、delay_ms、transmission_rate_mbps的字典

        Returns:
            归一化的状态向量
        """
        # 归一化
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
            动作索引
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            action = np.random.randint(0, self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.network.forward(state)
            action = np.argmax(q_values[0])

        return action

    def action_to_params(self, action: int) -> Dict[str, int]:
        """
        将动作索引转换为协议参数

        Args:
            action: 动作索引 (0-15)

        Returns:
            协议参数字典
        """
        # 16个动作 = 4 bundle_size × 4 ltp_block_size
        bundle_idx = action // 4
        block_idx = action % 4

        params = {
            "bundle_size": self.action_space["bundle_size"][bundle_idx],
            "ltp_block_size": self.action_space["ltp_block_size"][block_idx],
            "ltp_segment_size": self.action_space["ltp_segment_size"][block_idx],
            "session_count": self.action_space["session_count"][bundle_idx]
        }

        return params

    def optimize_params(self, state: Dict[str, float]) -> Dict[str, int]:
        """
        生成优化参数

        Args:
            state: 网络状态

        Returns:
            优化的协议参数
        """
        # 转换状态
        state_vector = self.discretize_state(state)

        # 选择动作
        action = self.select_action(state_vector, training=True)

        # 转换为参数
        params = self.action_to_params(action)

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
        """
        使用经验回放缓冲区中的一批经验进行训练
        """
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
        print(f"\n[DQN训练] 开始使用 {len(records)} 条记录进行批量训练")

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

                # 查找对应的动作（根据输出参数）
                action = self._find_action_from_params(output_data)

                # 存储经验
                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,  # 简化：使用相同的状态作为下一状态
                    done=False
                )

                # 进行训练
                loss = self.train_batch()

                if i % 10 == 0:
                    print(f"  [记录{i+1}/{len(records)}] 奖励: {reward:.4f}, 交付时间: {delivery_time_ms:.2f}ms, "
                          f"Loss: {loss:.6f}")

                self.episode_rewards.append(reward)

            except Exception as e:
                print(f"  [错误] 处理记录 {i} 失败: {e}")

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 增加模型版本
        self.model_version += 1

        # 输出统计信息
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        print(f"[DQN训练完成] 模型版本: {self.model_version}, "
              f"平均奖励: {avg_reward:.4f}, "
              f"探索率ε: {self.epsilon:.4f}, "
              f"训练步数: {self.training_steps}")

    def _find_action_from_params(self, params: Dict[str, int]) -> int:
        """
        从参数字典反向查找动作索引

        Args:
            params: 协议参数字典

        Returns:
            动作索引
        """
        try:
            bundle_idx = self.action_space["bundle_size"].index(params.get("bundle_size", 1024))
            block_idx = self.action_space["ltp_block_size"].index(params.get("ltp_block_size", 512))
            action = bundle_idx * 4 + block_idx
            return action
        except ValueError:
            # 如果找不到精确匹配，返回随机动作
            return np.random.randint(0, self.action_dim)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
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

            # 生成优化参数
            optimized_params = self.dqn_optimizer.optimize_params(link_state)

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

                    print(f"[参数响应] 已发送优化参数: {optimized_params}")

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
        print("DQN优化器服务器启动")
        print(f"参数请求服务器: 端口 {self.param_request_port}")
        print(f"记录接收服务器: 端口 {self.record_receive_port}")
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

    parser = argparse.ArgumentParser(description='BP/LTP DQN优化器')
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
