#!/usr/bin/env python3
"""
优化器/算力电脑C (Optimizer Computer C) - DQN v2.3 GPU版本
功能：
1. 接收节点A的优化请求
2. 基于DQN强化学习模型生成优化参数（Bundle、Block）
3. 自适应选择Segment大小（基于网络状态）
4. 自动计算LTP会话数量（不是训练得出）
5. 将优化参数发送给节点A
6. 接收节点B的训练记录
7. 使用训练记录进行DQN模型的训练和更新

关键改进（v2.3 GPU）：
- 使用PyTorch + CUDA GPU加速训练
- 快照机制（Copy-on-Write）解决并发问题
- 训练在GPU上进行，参数请求不阻塞
- 动作空间：32个采样的(bundle, block)组合
- segment_size：自适应选择（基于延时、误码率）
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
import copy

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"[GPU] 检测到CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA版本: {torch.version.cuda}")
        print(f"[GPU] PyTorch版本: {torch.__version__}")
    else:
        DEVICE = torch.device("cpu")
        print("[警告] CUDA不可用，将使用CPU训练")
except ImportError:
    print("[错误] 未安装PyTorch，请安装: pip install torch")
    TORCH_AVAILABLE = False
    DEVICE = None


def calculate_ltp_sessions(delay: float, bundle_size: int, file_size: int,
                           block_size: int, trans_rate: float) -> int:
    """计算LTP会话数量（与原版本相同）"""
    total_bundles = math.ceil(file_size / bundle_size)
    bundles_per_block = math.ceil(block_size / bundle_size)
    ltp_blocks = math.ceil(total_bundles / bundles_per_block)
    times = delay/500 + ((block_size + 20) / trans_rate)
    ltp_sessions = math.ceil((times * trans_rate) / (block_size + 20)) + 1
    ltp_sessions = min(ltp_sessions, ltp_blocks + 1, 20)
    return ltp_sessions


class ExperienceReplay:
    """经验回放缓冲区"""

    def __init__(self, max_size: int = 10000):
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience: Tuple):
        self.memory.append(experience)

    def sample_batch(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def is_full(self) -> bool:
        return len(self.memory) >= self.max_size


class DQNNetworkGPU(nn.Module):
    """DQN神经网络 - PyTorch + GPU实现"""

    def __init__(self, state_dim: int = 4, action_dim: int = 32, hidden_dim: int = 128):
        super(DQNNetworkGPU, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 定义网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        print(f"[DQN网络] 网络结构: {state_dim} -> {hidden_dim} -> {hidden_dim} -> {action_dim}")

    def forward(self, state):
        """前向传播"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class RewardCalculator:
    """奖励函数计算器（与原版本相同）"""

    def __init__(self, max_delivery_time_ms: float = 5000.0, min_delivery_time_ms: float = 100.0):
        self.max_delivery_time = max_delivery_time_ms
        self.min_delivery_time = min_delivery_time_ms

    def calculate_reward(self, delivery_time_ms: float, data_size: int,
                        bit_error_rate: float, delay_ms: float) -> float:
        time_reward = self._calculate_time_reward(delivery_time_ms)
        throughput_reward = self._calculate_throughput_reward(data_size, delivery_time_ms)
        robustness_reward = self._calculate_robustness_reward(bit_error_rate, delay_ms, delivery_time_ms)

        total_reward = (0.5 * time_reward + 0.3 * throughput_reward + 0.2 * robustness_reward)
        return total_reward

    def _calculate_time_reward(self, delivery_time_ms: float) -> float:
        if delivery_time_ms <= self.min_delivery_time:
            return 1.0
        elif delivery_time_ms >= self.max_delivery_time:
            return -1.0
        else:
            time_ratio = (delivery_time_ms - self.min_delivery_time) / \
                         (self.max_delivery_time - self.min_delivery_time)
            return 1.0 - 2.0 * time_ratio

    def _calculate_throughput_reward(self, data_size: int, delivery_time_ms: float) -> float:
        throughput_mbps = (data_size * 8) / (delivery_time_ms / 1000) / 1_000_000
        max_expected_throughput = 100.0
        throughput_normalized = min(throughput_mbps / max_expected_throughput, 1.0)
        return 2.0 * throughput_normalized - 1.0

    def _calculate_robustness_reward(self, bit_error_rate: float, delay_ms: float,
                                     delivery_time_ms: float) -> float:
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


class DQNOptimizerGPU:
    """DQN优化器 - GPU加速 + 快照机制"""

    def __init__(self, device=None, pretrained_model: str = None):
        """初始化DQN优化器（GPU版本）

        Args:
            device: 训练设备（cuda或cpu）
            pretrained_model: 预训练模型路径（可选）
        """
        self.state_dim = 4
        self.device = device if device else DEVICE

        # 动作空间
        all_valid_combinations = self._generate_all_valid_combinations()
        self.valid_action_pairs = self._sample_32_combinations(all_valid_combinations)
        self.action_dim = len(self.valid_action_pairs)

        # segment选项
        self.segment_options = [200, 400, 600, 800, 1000, 1200, 1400]

        # 初始化网络（在GPU上）
        self.policy_net = DQNNetworkGPU(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetworkGPU(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设为评估模式

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # 经验回放
        self.experience_replay = ExperienceReplay(max_size=10000)

        # 奖励计算器
        self.reward_calculator = RewardCalculator()

        # 训练参数
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 32

        # 统计信息
        self.model_version = 0
        self.training_steps = 0
        self.episode_rewards = deque(maxlen=100)

        # 快照机制锁
        self.snapshot_lock = threading.Lock()
        self.inference_net = None  # 用于推理的快照网络

        print(f"[DQN优化器v2.3-GPU] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if self.device.type == 'cuda' else "")
        print(f"  Bundle大小覆盖: {len(set([b for b, _ in self.valid_action_pairs]))}种")
        print(f"  Block大小覆盖: {len(set([bl for _, bl in self.valid_action_pairs]))}种")

        # 加载预训练模型（如果提供）
        if pretrained_model:
            import os
            if os.path.exists(pretrained_model):
                self.load_pretrained_model(pretrained_model)
            else:
                print(f"[警告] 预训练模型文件不存在: {pretrained_model}")

        # 创建推理快照（在加载预训练模型之后，或使用随机初始化）
        self._create_inference_snapshot()

    def _create_inference_snapshot(self):
        """创建推理网络快照（CPU上）"""
        with self.snapshot_lock:
            self.inference_net = DQNNetworkGPU(self.state_dim, self.action_dim).to('cpu')

            # 复制当前策略网络的权重到CPU
            cpu_state_dict = {k: v.cpu() for k, v in self.policy_net.state_dict().items()}
            self.inference_net.load_state_dict(cpu_state_dict)
            self.inference_net.eval()

            print(f"[快照] 已创建推理网络快照（版本{self.model_version}）")

    def load_pretrained_model(self, model_path: str):
        """加载预训练模型

        Args:
            model_path: 模型文件路径
        """
        try:
            print(f"[预训练] 正在加载模型: {model_path}")

            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # 加载网络权重
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.target_net.eval()

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练状态
            self.model_version = checkpoint.get('model_version', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            self.epsilon = checkpoint.get('epsilon', 0.1)

            # 加载历史奖励（如果有）
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)

            print(f"[预训练] ✅ 已加载预训练模型: {model_path}")
            print(f"  • 模型版本: {self.model_version}")
            print(f"  • 训练步数: {self.training_steps}")
            print(f"  • 探索率: {self.epsilon:.4f}")
            print(f"  • 历史奖励数: {len(self.episode_rewards)}")
            print(f"  → 推理快照将在初始化完成后创建")

        except Exception as e:
            print(f"[警告] 加载预训练模型失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_all_valid_combinations(self) -> List[Tuple[int, int]]:
        """生成所有满足约束的(bundle_size, block_size)组合"""
        bundle_sizes = [
            1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000,
            20000, 24000, 30000, 40000, 60000, 80000, 100000
        ]
        block_sizes = [
            20000, 40000, 60000, 80000, 100000, 120000, 140000,
            160000, 180000, 200000, 220000, 240000, 260000, 280000,
            300000, 350000, 400000, 450000, 500000, 1000000
        ]

        valid_combinations = []
        for bundle in bundle_sizes:
            for block in block_sizes:
                if block >= bundle and block % bundle == 0:
                    valid_combinations.append((bundle, block))

        return valid_combinations

    def _sample_32_combinations(self, all_combinations: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """从所有有效组合中采样32个代表性组合"""
        from collections import defaultdict

        by_bundle = defaultdict(list)
        for bundle, block in all_combinations:
            by_bundle[bundle].append(block)

        sampled = []
        for bundle in sorted(by_bundle.keys()):
            blocks = sorted(by_bundle[bundle])

            if len(blocks) == 1:
                sampled.append((bundle, blocks[0]))
            elif len(blocks) == 2:
                sampled.extend([(bundle, b) for b in blocks])
            else:
                sampled.append((bundle, blocks[0]))
                sampled.append((bundle, blocks[len(blocks)//2]))
                sampled.append((bundle, blocks[-1]))

        return sampled[:32]

    def _adaptive_segment_size(self, delay_ms: float, bit_error_rate: float, block_size: int) -> int:
        """根据网络状态自适应选择segment_size"""
        delay_factor = min(delay_ms / 1000.0, 1.0)
        ber_factor = min(bit_error_rate / 0.01, 1.0)
        adversity = 0.6 * delay_factor + 0.4 * ber_factor

        if adversity > 0.7:
            segment_size = 1400
        elif adversity > 0.5:
            segment_size = 1000
        elif adversity > 0.3:
            segment_size = 600
        else:
            segment_size = 200

        return segment_size

    def _validate_segment_size(self, segment_size: int, block_size: int) -> int:
        """验证segment_size是否合理"""
        max_segment = int(block_size * 0.5)

        if segment_size > max_segment:
            valid_segments = [s for s in self.segment_options if s <= max_segment]
            if valid_segments:
                return max(valid_segments)
            else:
                return 200

        return segment_size

    def discretize_state(self, state: Dict[str, float]) -> np.ndarray:
        """将连续状态转换为归一化向量"""
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
        """使用ε-贪心策略选择动作（使用快照网络，不阻塞）"""
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            # 使用快照网络进行推理（在CPU上，不阻塞GPU训练）
            with torch.no_grad():
                with self.snapshot_lock:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.inference_net(state_tensor)
                    action = q_values.argmax().item()

        return action

    def action_to_params(self, action: int, data_size: int, delay_ms: float,
                        transmission_rate_mbps: float, bit_error_rate: float = 1e-5) -> Dict[str, int]:
        """将动作索引转换为协议参数"""
        if action >= self.action_dim:
            raise ValueError(f"动作{action}超出范围[0, {self.action_dim-1}]")

        bundle_size, ltp_block_size = self.valid_action_pairs[action]

        ltp_segment_size = self._adaptive_segment_size(delay_ms, bit_error_rate, ltp_block_size)
        ltp_segment_size = self._validate_segment_size(ltp_segment_size, ltp_block_size)

        trans_rate_bytes = transmission_rate_mbps * 1_000_000 / 8
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
            "session_count": ltp_sessions
        }

        return params

    def optimize_params(self, request_data: Dict[str, Any]) -> Dict[str, int]:
        """生成优化参数（无阻塞，使用快照）"""
        data_size = request_data.get("data_size", 10240)
        link_state = request_data.get("link_state", {})

        state = {
            "data_size": data_size,
            "bit_error_rate": link_state.get("bit_error_rate", 1e-5),
            "delay_ms": link_state.get("delay_ms", 100.0),
            "transmission_rate_mbps": link_state.get("transmission_rate_mbps", 10.0)
        }

        state_vector = self.discretize_state(state)
        action = self.select_action(state_vector, training=True)

        params = self.action_to_params(
            action=action,
            data_size=data_size,
            delay_ms=state["delay_ms"],
            transmission_rate_mbps=state["transmission_rate_mbps"],
            bit_error_rate=state["bit_error_rate"]
        )

        print(f"[DQN优化-GPU] 动作={action}/{self.action_dim}, "
              f"Bundle={params['bundle_size']}, Block={params['ltp_block_size']}, "
              f"Segment={params['ltp_segment_size']}(自适应)")

        return params

    def store_experience(self, state: Dict[str, float], action: int, reward: float,
                        next_state: Dict[str, float], done: bool):
        """存储经验到回放缓冲区"""
        state_vector = self.discretize_state(state)
        next_state_vector = self.discretize_state(next_state)
        self.experience_replay.add((state_vector, action, reward, next_state_vector, done))

    def train_batch(self):
        """使用GPU训练一批经验"""
        if len(self.experience_replay.memory) < self.batch_size:
            return 0.0

        batch = self.experience_replay.sample_batch(self.batch_size)

        # 准备批次数据并移到GPU
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        # 软更新目标网络
        if self.training_steps % 100 == 0:
            self._soft_update_target_network(tau=0.001)

        return loss.item()

    def _soft_update_target_network(self, tau: float = 0.001):
        """软更新目标网络"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def batch_update_model(self, records: List[Dict[str, Any]]):
        """使用一批训练记录更新模型（GPU训练）"""
        print(f"\n[DQN训练-GPU] 开始使用 {len(records)} 条记录进行批量训练")
        print(f"[GPU] 训练设备: {self.device}")

        start_time = time.time()

        for i, record in enumerate(records):
            try:
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

                reward = self.reward_calculator.calculate_reward(
                    delivery_time_ms=delivery_time_ms,
                    data_size=input_data.get("data_size", 0),
                    bit_error_rate=input_data.get("bit_error_rate", 0),
                    delay_ms=input_data.get("delay_ms", 0)
                )

                action = self._find_action_from_params(output_data)

                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,
                    done=False
                )

                loss = self.train_batch()

                if i % 10 == 0:
                    print(f"  [记录{i+1}/{len(records)}] 奖励: {reward:.4f}, "
                          f"交付时间: {delivery_time_ms:.2f}ms, Loss: {loss:.6f}")

                self.episode_rewards.append(reward)

            except Exception as e:
                print(f"  [错误] 处理记录 {i} 失败: {e}")

        # 训练完成后更新推理快照
        self._create_inference_snapshot()

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.model_version += 1

        elapsed = time.time() - start_time
        avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0

        print(f"[DQN训练完成-GPU] 模型版本: {self.model_version}, "
              f"平均奖励: {avg_reward:.4f}, 探索率ε: {self.epsilon:.4f}, "
              f"训练步数: {self.training_steps}, 耗时: {elapsed:.2f}s")

    def _find_action_from_params(self, params: Dict[str, int]) -> int:
        """从参数字典反向查找动作索引"""
        bundle = params.get("bundle_size", 1024)
        block = params.get("ltp_block_size", 512)

        try:
            for idx, (b, bl) in enumerate(self.valid_action_pairs):
                if b == bundle and bl == block:
                    return idx

            for idx, (b, bl) in enumerate(self.valid_action_pairs):
                if b == bundle:
                    return idx

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
            "avg_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0,
            "device": str(self.device)
        }


class OptimizerServer:
    """优化器服务器 - GPU版本"""

    def __init__(self, param_request_port: int = 5002, record_receive_port: int = 5003, pretrained_model: str = None):
        """初始化优化器服务器

        Args:
            param_request_port: 参数请求端口
            record_receive_port: 记录接收端口
            pretrained_model: 预训练模型路径（可选）
        """
        self.param_request_port = param_request_port
        self.record_receive_port = record_receive_port

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch未安装，无法启动GPU优化器")

        self.dqn_optimizer = DQNOptimizerGPU(pretrained_model=pretrained_model)
        self.running = True

    def handle_param_request(self, request_data: Dict[str, Any]) -> Dict[str, int]:
        """处理参数优化请求（无阻塞）"""
        try:
            link_state = request_data.get("link_state", {})

            print(f"[参数请求-GPU] 数据量: {request_data.get('data_size')} bytes, "
                  f"误码率: {link_state.get('bit_error_rate')}, "
                  f"延时: {link_state.get('delay_ms')}ms")

            # 使用快照网络，不阻塞GPU训练
            optimized_params = self.dqn_optimizer.optimize_params(request_data)

            return optimized_params

        except Exception as e:
            print(f"[错误] 处理参数请求失败: {e}")
            return {}

    def param_request_server(self):
        """处理参数请求的服务器线程"""
        print(f"[参数请求服务器-GPU] 启动 (监听端口 {self.param_request_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.param_request_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

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

                    request = json.loads(request_data.decode('utf-8'))
                    print(f"\n[新请求-GPU] 来自 {client_address}")

                    optimized_params = self.handle_param_request(request)

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

                    print(f"[参数响应-GPU] 已发送优化参数")

                    client_socket.close()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[错误] 处理参数请求失败: {e}")

        finally:
            server_socket.close()

    def record_receive_server(self):
        """处理训练记录接收的服务器线程"""
        print(f"[记录接收服务器-GPU] 启动 (监听端口 {self.record_receive_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.record_receive_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

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

                    message = json.loads(message_data.decode('utf-8'))
                    message_type = message.get("type")

                    if message_type == "training_records":
                        records = message.get("records", [])
                        print(f"\n[收到训练记录-GPU] {len(records)} 条来自 {client_address}")

                        # GPU训练（不阻塞参数请求）
                        self.dqn_optimizer.batch_update_model(records)

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
        print("DQN优化器服务器v2.3-GPU启动")
        print(f"参数请求服务器: 端口 {self.param_request_port}")
        print(f"记录接收服务器: 端口 {self.record_receive_port}")
        print("关键特性: ")
        print("  - NVIDIA GPU加速训练（RTX 4060）")
        print("  - 快照机制（Copy-on-Write）")
        print("  - 参数请求无阻塞")
        print("  - 32个采样动作空间")
        print("  - segment自适应选择")
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

        param_thread.join(timeout=5)
        record_thread.join(timeout=5)


def main():
    """主函数"""
    # ==================== 配置参数（在此修改） ====================
    param_request_port = 5002          # 参数请求端口
    record_receive_port = 5003         # 训练记录端口
    pretrained_model = None            # 预训练模型路径（None表示不使用）

    # 如果要使用预训练模型，取消下面一行的注释并修改路径：
    # pretrained_model = '/root/agent/computer/dqn_model_pretrained.pth'
    # ============================================================

    print("="*60)
    print("BP/LTP DQN优化器v2.3-GPU")
    print("="*60)
    print(f"参数请求端口: {param_request_port}")
    print(f"训练记录端口: {record_receive_port}")
    print(f"预训练模型: {pretrained_model if pretrained_model else '不使用（随机初始化）'}")
    print("="*60)
    print()

    optimizer = OptimizerServer(
        param_request_port=param_request_port,
        record_receive_port=record_receive_port,
        pretrained_model=pretrained_model
    )

    optimizer.run()


if __name__ == "__main__":
    main()