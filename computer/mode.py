#!/usr/bin/env python3
"""
优化器/算力电脑C (Optimizer Computer C)
功能：
1. 接收节点A的优化请求
2. 基于强化学习模型生成优化参数
3. 将优化参数发送给节点A
4. 接收节点B的训练记录
5. 使用训练记录进行强化学习模型的训练和更新
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


class SimpleRLOptimizer:
    """
    简化的强化学习优化器
    这是一个基础框架，可以根据需要集成更复杂的RL算法（DQN、PPO等）
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 4):
        """
        初始化强化学习优化器

        Args:
            state_dim: 状态维度（数据量、误码率、延时、速率）
            action_dim: 动作维度（Bundle、Block、Segment、会话数）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 简化的Q表（实际应用中应使用神经网络）
        # 这里使用状态空间离散化和Q学习
        self.state_bins = [10, 10, 10, 10]  # 每个状态维度的离散化级数
        self.action_space = {
            "bundle_size": [512, 1024, 2048, 4096],
            "ltp_block_size": [256, 512, 1024, 2048],
            "ltp_segment_size": [128, 256, 512, 1024],
            "session_count": [1, 2, 4, 8]
        }

        # 初始化Q表
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 探索率

        # 性能记录
        self.training_records = deque(maxlen=1000)
        self.model_version = 0

    def discretize_state(self, state: Dict[str, float]) -> Tuple[int, ...]:
        """
        将连续状态转换为离散状态

        Args:
            state: 包含data_size、bit_error_rate、delay_ms、transmission_rate_mbps的字典

        Returns:
            离散化后的状态元组
        """
        # 归一化和离散化
        data_size_normalized = min(state.get("data_size", 0) / 100000, 1.0)
        ber_normalized = min(state.get("bit_error_rate", 0) * 1e6, 1.0)
        delay_normalized = min(state.get("delay_ms", 0) / 500, 1.0)
        rate_normalized = min(state.get("transmission_rate_mbps", 0) / 100, 1.0)

        state_normalized = [
            data_size_normalized,
            ber_normalized,
            delay_normalized,
            rate_normalized
        ]

        # 离散化
        discretized_state = tuple(
            min(int(s * self.state_bins[i]), self.state_bins[i] - 1)
            for i, s in enumerate(state_normalized)
        )

        return discretized_state

    def select_action(self, state: Tuple[int, ...]) -> int:
        """
        基于ε-贪心策略选择动作

        Args:
            state: 离散化的状态

        Returns:
            动作索引
        """
        # 获取该状态的Q值
        if state not in self.q_table:
            self.q_table[state] = np.zeros(16)  # 4x4=16个可能的组合

        q_values = self.q_table[state]

        # ε-贪心选择
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            action = np.random.randint(0, 16)
        else:
            # 利用：选择最优动作
            action = np.argmax(q_values)

        return action

    def action_to_params(self, action: int) -> Dict[str, int]:
        """
        将动作索引转换为协议参数

        Args:
            action: 动作索引 (0-15)

        Returns:
            协议参数字典
        """
        # 4种bundle大小 x 4种block大小 = 16个组合
        bundle_idx = action // 4
        block_idx = action % 4

        params = {
            "bundle_size": self.action_space["bundle_size"][bundle_idx],
            "ltp_block_size": self.action_space["ltp_block_size"][block_idx],
            "ltp_segment_size": self.action_space["ltp_segment_size"][block_idx],
            "session_count": self.action_space["session_count"][bundle_idx % 4]
        }

        return params

    def optimize_params(self, state: Dict[str, float]) -> Dict[str, int]:
        """
        根据状态生成优化后的参数

        Args:
            state: 当前网络状态

        Returns:
            优化后的协议参数
        """
        # 离散化状态
        discretized_state = self.discretize_state(state)

        # 选择动作
        action = self.select_action(discretized_state)

        # 转换为参数
        params = self.action_to_params(action)

        print(f"[优化器] 选择动作 {action}, 参数: {params}")

        return params

    def update_model(self, record: Dict[str, Any]):
        """
        使用单条训练记录更新模型

        Args:
            record: 包含input、output、performance的训练记录
        """
        try:
            # 提取信息
            input_data = record.get("input", {})
            output_data = record.get("output", {})
            performance = record.get("performance", {})

            # 计算奖励（反向：我们想最小化交付时间）
            delivery_time_ms = performance.get("delivery_time_ms", float('inf'))
            reward = -delivery_time_ms / 1000  # 转换为秒，取反

            # 创建状态
            state = {
                "data_size": input_data.get("data_size", 0),
                "bit_error_rate": input_data.get("bit_error_rate", 0),
                "delay_ms": input_data.get("delay_ms", 0),
                "transmission_rate_mbps": input_data.get("transmission_rate_mbps", 0)
            }

            # 离散化状态
            discretized_state = self.discretize_state(state)

            # 查找对应的动作
            # 这里简化处理：从输出参数反向查找动作
            bundle_idx = self.action_space["bundle_size"].index(output_data.get("bundle_size", 1024))
            block_idx = self.action_space["ltp_block_size"].index(output_data.get("ltp_block_size", 512))
            action = bundle_idx * 4 + block_idx

            # 初始化Q表项
            if discretized_state not in self.q_table:
                self.q_table[discretized_state] = np.zeros(16)

            # Q学习更新
            current_q = self.q_table[discretized_state][action]
            max_next_q = np.max(self.q_table.get(discretized_state, np.zeros(16)))

            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[discretized_state][action] = new_q

            print(f"[模型更新] 状态: {discretized_state}, 动作: {action}, "
                  f"奖励: {reward:.4f}, Q值: {new_q:.4f}, 交付时间: {delivery_time_ms:.2f}ms")

        except Exception as e:
            print(f"[错误] 模型更新失败: {e}")

    def batch_update_model(self, records: List[Dict[str, Any]]):
        """
        批量更新模型

        Args:
            records: 训练记录列表
        """
        print(f"\n[批量更新] 开始使用 {len(records)} 条记录进行训练")

        for record in records:
            self.update_model(record)

        self.model_version += 1
        print(f"[模型更新完成] 模型版本: {self.model_version}")


class OptimizerServer:
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

        self.rl_optimizer = SimpleRLOptimizer()
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
            data_size = request_data.get("data_size")
            link_state = request_data.get("link_state", {})
            current_params = request_data.get("current_params", {})

            print(f"[参数请求] 数据量: {data_size}, 链路状态: {link_state}")

            # 生成优化参数
            optimized_params = self.rl_optimizer.optimize_params(link_state)

            return optimized_params

        except Exception as e:
            print(f"[错误] 处理参数请求失败: {e}")
            return request_data.get("current_params", {})

    def param_request_server(self):
        """
        处理参数请求的服务器线程
        """
        print(f"[参数请求服务器] 启动 (监听端口 {self.param_request_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.param_request_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

                    # 接收请求长度
                    length_data = client_socket.recv(4)
                    if not length_data:
                        continue

                    message_length = struct.unpack('!I', length_data)[0]

                    # 接收请求内容
                    request_data = b''
                    while len(request_data) < message_length:
                        chunk = client_socket.recv(min(4096, message_length - len(request_data)))
                        if not chunk:
                            break
                        request_data += chunk

                    # 解析请求
                    request = json.loads(request_data.decode('utf-8'))
                    print(f"\n[新请求] 来自 {client_address}")

                    # 处理请求
                    optimized_params = self.handle_param_request(request)

                    # 发送响应
                    response = {
                        "status": "success",
                        "optimized_params": optimized_params,
                        "model_version": self.rl_optimizer.model_version,
                        "timestamp": time.time()
                    }

                    response_json = json.dumps(response)
                    response_message = response_json.encode('utf-8')

                    client_socket.sendall(struct.pack('!I', len(response_message)))
                    client_socket.sendall(response_message)

                    client_socket.close()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[错误] 处理参数请求失败: {e}")

        finally:
            server_socket.close()

    def record_receive_server(self):
        """
        处理训练记录接收的服务器线程
        """
        print(f"[记录接收服务器] 启动 (监听端口 {self.record_receive_port})")

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.record_receive_port))
            server_socket.listen(5)

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

                    # 接收消息长度
                    length_data = client_socket.recv(4)
                    if not length_data:
                        continue

                    message_length = struct.unpack('!I', length_data)[0]

                    # 接收消息内容
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
                        self.rl_optimizer.batch_update_model(records)

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
        """
        运行优化器服务器
        """
        print("="*60)
        print("优化器(算力电脑C)启动")
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

    parser = argparse.ArgumentParser(description='BP/LTP强化学习优化器(电脑C)')
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
