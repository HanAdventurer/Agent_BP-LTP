#!/usr/bin/env python3
"""
接收节点B (Receiver Node B)
功能：
1. 接收节点A的数据传输
2. 记录接收完成时间
3. 计算业务交付时间
4. 通过记录器模块生成训练记录
5. 周期性地将记录发送到电脑C
"""

import socket
import json
import time
import random
import struct
import threading
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import deque
import math

# 导入BP/LTP接收器接口
try:
    from bp_ltp_receiver_interface import BPLTPReceiverInterface
    BP_LTP_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入BP/LTP接收器接口，将使用模拟模式")
    BP_LTP_AVAILABLE = False


class RecordLogger:
    """记录器模块 - 用于记录自适应优化的输入输出和性能表现"""

    def __init__(self, buffer_size: int = 100, flush_interval: int = 6000, csv_file: str = None):
        """
        初始化记录器

        Args:
            buffer_size: 缓冲区大小（记录数）
            flush_interval: 刷新间隔（秒）
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.records = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.last_flush_time = time.time()
        self.csv_file = csv_file
                # 如果指定了CSV文件，初始化CSV文件（写入表头）
        if self.csv_file:
            self._initialize_csv()

    def add_record(self, record: Dict[str, Any]):
        """
        添加一条记录

        Args:
            record: 包含输入、输出和性能的记录
        """
        with self.lock:
            self.records.append(record)
            print(f"[记录器] 添加新记录 (当前缓冲数: {len(self.records)}/{self.buffer_size})")

            # 同步保存到CSV文件
            if self.csv_file:
                self._save_to_csv(record)
    def _initialize_csv(self):
        """
        初始化CSV文件，写入表头
        """
        import csv
        import os

        try:
            # 检查文件是否存在
            file_exists = os.path.exists(self.csv_file)

            if not file_exists:
                # 创建CSV文件并写入表头
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'data_size',
                        'bit_error_rate',
                        'delay_ms',
                        'transmission_rate_mbps',
                        'bundle_size',
                        'ltp_block_size',
                        'ltp_segment_size',
                        'session_count',
                        'delivery_time_ms',
                        'throughput_mbps'
                    ])
                print(f"[记录器] CSV文件已创建: {self.csv_file}")
            else:
                print(f"[记录器] CSV文件已存在: {self.csv_file}")

        except Exception as e:
            print(f"[错误] 初始化CSV文件失败: {e}")

    def _save_to_csv(self, record: Dict[str, Any]):
        """
        将单条记录同步保存到CSV文件

        Args:
            record: 训练记录
        """
        if not self.csv_file:
            return

        import csv

        try:
            # 提取记录字段
            input_data = record.get("input", {})
            output_data = record.get("output", {})
            performance_data = record.get("performance", {})
            timestamp = record.get("timestamp", time.time())

            data_size = input_data.get("data_size", 0)
            delivery_time_ms = performance_data.get("delivery_time_ms", 0)

            # 计算吞吐量（Mbps）
            if delivery_time_ms > 0:
                # delivery_time_ms是毫秒，需要转换为秒来计算吞吐量
                throughput_mbps = (data_size * 8 / 1_000_000) / (delivery_time_ms / 1000)
            else:
                throughput_mbps = 0.0

            # 追加写入CSV文件
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    data_size,
                    input_data.get("bit_error_rate", 0),
                    input_data.get("delay_ms", 0),
                    input_data.get("transmission_rate_mbps", 0),
                    output_data.get("bundle_size", 0),
                    output_data.get("ltp_block_size", 0),
                    output_data.get("ltp_segment_size", 0),
                    output_data.get("session_count", 0),
                    round(delivery_time_ms, 3),
                    throughput_mbps
                ])

        except Exception as e:
            print(f"[错误] 保存记录到CSV失败: {e}")

    def should_flush(self) -> bool:
        """
        检查是否应该刷新记录

        Returns:
            True 如果缓冲区满或刷新间隔已过期
        """
        with self.lock:
            buffer_full = len(self.records) >= self.buffer_size
            time_expired = (time.time() - self.last_flush_time) >= self.flush_interval

            return buffer_full or time_expired

    def get_records_to_send(self) -> List[Dict[str, Any]]:
        """
        获取待发送的记录

        Returns:
            记录列表
        """
        with self.lock:
            records_to_send = list(self.records)
            self.records.clear()
            self.last_flush_time = time.time()
            return records_to_send

    def record_transmission(self,
                           data_size: int,
                           bit_error_rate: float,
                           delay_ms: float,
                           transmission_rate_mbps: float,
                           bundle_size: int,
                           ltp_block_size: int,
                           ltp_segment_size: int,
                           session_count: int,
                           delivery_time_ms: float):
        """
        记录一次传输的完整信息

        Args:
            data_size: 数据量大小
            bit_error_rate: 误码率
            delay_ms: 延时（ms）
            transmission_rate_mbps: 传输速率（Mbps）
            bundle_size: Bundle大小
            ltp_block_size: LTP Block大小
            ltp_segment_size: LTP Segment大小
            session_count: 会话数量
            delivery_time_ms: 业务交付时间（毫秒）
        """
        record = {
            "input": {
                "data_size": data_size,
                "bit_error_rate": bit_error_rate,
                "delay_ms": delay_ms,
                "transmission_rate_mbps": transmission_rate_mbps
            },
            "output": {
                "bundle_size": bundle_size,
                "ltp_block_size": ltp_block_size,
                "ltp_segment_size": ltp_segment_size,
                "session_count": session_count
            },
            "performance": {
                "delivery_time_ms": delivery_time_ms
            },
            "timestamp": time.time()
        }

        self.add_record(record)


class ReceiverNode:
    def __init__(self,
                 listen_port: int = 5001,
                 optimizer_host: str = "192.168.1.3",
                 optimizer_port: int = 5003,
                 own_eid_number: int = 2,
                 use_bp_ltp: bool = True, 
                 csv_file: str = "receiver_records.csv",
                 sender_host: str = "192.168.137.194",
                 sender_notification_port: int = 5009):
        """
        初始化接收节点

        Args:
            listen_port: 监听端口（用于接收元数据）
            optimizer_host: 优化器(电脑C)的IP地址
            optimizer_port: 优化器的数据接收端口
            own_eid_number: 本节点EID数字（例如 2）
            use_bp_ltp: 是否使用BP/LTP协议栈接收
        """
        self.listen_port = listen_port
        self.optimizer_host = optimizer_host
        self.optimizer_port = optimizer_port
        self.own_eid_number = own_eid_number
        self.sender_host = sender_host
        self.sender_notification_port = sender_notification_port

        self.logger = RecordLogger(csv_file=csv_file)
        self.running = True

        # 链路配置锁 - 防止并发处理多个链路配置请求
        self.link_config_lock = threading.Lock()
        self.link_config_processing = False

        # 消息去重：记录已处理的transmission_id（保留最近100个）
        self.processed_transmissions = set()
        self.max_processed_history = 100

        # 用于存储当前传输的元数据
        self.current_transmission = {
            "transmission_id": None,  # 新增：保存当前传输的ID
            "data_size": 0,
            "start_timestamp": 0.0,
            "link_state": {},
            "protocol_params": {},
            "bundle_size": 1000
        }

        # 接收状态追踪
        self.reception_thread = None
        self.reception_event = threading.Event()
        self.reception_result = {
            "stop_time": 0.0,
            "report": "",
            "success": False
        }

        # 初始化BP/LTP接收器接口
        self.use_bp_ltp = use_bp_ltp and BP_LTP_AVAILABLE
        self.bp_ltp_receiver = None

        if self.use_bp_ltp:
            try:
                self.bp_ltp_receiver = BPLTPReceiverInterface(
                    own_eid_number=own_eid_number
                )
                print(f"[初始化] BP/LTP接收器接口已启用")
            except Exception as e:
                print(f"[警告] 初始化BP/LTP接收器接口失败: {e}")
                self.use_bp_ltp = False

    def notify_sender_reception_complete(self) -> bool:
        """
        通知发送端接收已完成

        注意：通知消息中包含 transmission_id，用于发送端验证，防止消息串扰

        Returns:
            通知是否成功
        """
        # 获取当前传输的 transmission_id
        transmission_id = self.current_transmission.get("transmission_id")

        notification = {
            "type": "reception_complete",
            "transmission_id": transmission_id,  # 新增：包含传输ID
            "timestamp": time.time()
        }

        print(f"[通知] 准备通知发送端接收完成 (transmission_id={transmission_id})")
        notification_json = json.dumps(notification).encode('utf-8')
        message_len = struct.pack('!I', len(notification_json))

        attempt = 0
        backoff = 2.0  # 初始退避时间增加到2秒
        max_backoff = 60.0  # 最大退避时间增加到60秒
        max_attempts = 15  # ✅ 增加到15次重试（应对时序竞态）

        while attempt < max_attempts:
            attempt += 1
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30.0)  # 增加timeout到30秒

                print(f"[通知] 第{attempt}次尝试连接发送端 {self.sender_host}:{self.sender_notification_port}")
                sock.connect((self.sender_host, self.sender_notification_port))

                sock.sendall(message_len)
                sock.sendall(notification_json)

                ack = sock.recv(1024)
                sock.close()

                if not ack:
                    print(f"[警告] 第{attempt}次发送未收到确认，准备重试")
                    continue

                try:
                    ack_text = ack.decode('utf-8', errors='ignore').strip()
                except Exception:
                    ack_text = ''

                # 如果收到常见确认字样则视为成功
                if ack_text.upper() in ("OK", "ACK", "RECEIVED") or len(ack_text) > 0:
                    print(f"[完成通知] ✅ 已成功通知发送端接收完成（第{attempt}次尝试），确认: {ack_text}")
                    return True
                else:
                    print(f"[警告] 第{attempt}次发送收到非确认信息: '{ack_text}'，准备重试")

            except KeyboardInterrupt:
                print("[中断] 用户中断重试，放弃发送完成通知")
                return False
            except OSError as e:
                # ✅ 不再特殊处理Connection refused
                # Connection refused可能是因为发送端监听socket还没准备好（时序问题）
                # 应该继续重试而不是立即放弃
                if "Connection refused" in str(e) or (hasattr(e, 'errno') and e.errno == 111):
                    print(f"[警告] 第{attempt}次连接被拒绝（Connection refused）")
                    print(f"  → 可能原因：发送端监听socket还没准备好")
                    print(f"  → 将在退避后重试")
                else:
                    print(f"[警告] 第{attempt}次发送完成通知失败: {e}")
            except Exception as e:
                print(f"[警告] 第{attempt}次发送完成通知失败: {e}")

            # 检查是否已达最大重试次数
            if attempt >= max_attempts:
                print(f"[通知失败] 已达最大重试次数{max_attempts}，放弃发送完成通知")
                return False

            # 指数退避并带随机抖动
            sleep_time = backoff + random.uniform(0, 0.5)
            print(f"[重试] 在 {sleep_time:.1f}s 后进行第{attempt+1}次尝试")
            time.sleep(sleep_time)
            backoff = min(max_backoff, backoff * 2)

        return False  # ✅ 超出最大重试次数
        
    def start_bp_ltp_reception(self, data_size: int, bundle_size: int) -> bool:
        """
        启动BP/LTP数据接收监听

        Args:
            data_size: 期望接收的总数据大小
            bundle_size: bundle大小

        Returns:
            是否成功启动
        """
        if not self.use_bp_ltp or not self.bp_ltp_receiver:
            print(f"[接收] BP/LTP接收器未启用，跳过BP/LTP接收")
            return False

        try:
            # 计算预期接收的bundle数量
            bundle_count = self.bp_ltp_receiver.calculate_bundle_count(data_size, bundle_size)

            # 在单独的线程中启动bpcounter监听
            self.reception_event.clear()
            self.reception_thread = threading.Thread(
                target=self._bp_ltp_reception_thread,
                args=(bundle_count,),
                daemon=True
            )
            self.reception_thread.start()

            print(f"[接收] BP/LTP接收监听线程已启动")
            return True

        except Exception as e:
            print(f"[错误] 启动BP/LTP接收失败: {e}")
            return False

    def _bp_ltp_reception_thread(self, bundle_count: int):
        """
        BP/LTP接收监听线程

        Args:
            bundle_count: 预期接收的bundle数量
        """
        try:
            print(f"[接收线程] 开始监听 {bundle_count} 个bundles")

            # 调用BP/LTP接收器接口监听数据
            report, stop_time = self.bp_ltp_receiver.monitor_reception(bundle_count)

            # 解析报告
            metrics = self.bp_ltp_receiver.parse_bpcounter_report(report)

            # 保存接收结果
            self.reception_result["stop_time"] = stop_time
            self.reception_result["report"] = report
            self.reception_result["metrics"] = metrics
            self.reception_result["success"] = True

            print(f"[接收线程] 接收完成，停止时间: {stop_time}")
            self.reception_event.set()
            # 立即通知发送端接收已完成
            self.notify_sender_reception_complete()

        except Exception as e:
            print(f"[错误] 接收线程异常: {e}")
            self.reception_result["success"] = False
            self.reception_event.set()

    def handle_link_config(self, data: Dict[str, Any]) -> bool:
        """
        处理链路配置请求 - 接收来自发送节点A的链路状态，并同步配置网络
        使用锁机制防止并发处理多个链路配置请求

        Args:
            data: 包含链路状态、数据大小和bundle大小的配置

        Returns:
            处理是否成功
        """
        # 尝试获取锁，如果已经在处理中，则直接返回成功（避免重复处理）
        if not self.link_config_lock.acquire(blocking=False):
            print(f"[链路配置] 已有链路配置正在处理中，忽略此次重复请求")
            return True

        try:
            self.link_config_processing = True
            print(f"[链路配置] 开始处理链路配置请求")

            link_state = data.get("link_state", {})
            data_size = data.get("data_size", 0)
            bundle_size = data.get("bundle_size", 1000)
            dest_addr = data.get("dest_addr", "192.168.1.1")  # 发送节点的地址
            sequence = data.get("sequence", 1)  # 获取sequence字段，用于EID配置
            transmission_id = data.get("transmission_id")  # 新增：获取传输ID

            # 清除接收事件，准备新的传输
            self.reception_event.clear()

            # 重置reception_result
            self.reception_result = {
                "stop_time": 0.0,
                "report": "",
                "success": False
            }

            # 重置current_transmission，保存transmission_id
            self.current_transmission = {
                "transmission_id": transmission_id,  # 新增：保存传输ID
                "data_size": 0,
                "start_timestamp": 0.0,
                "link_state": {},
                "protocol_params": {},
                "bundle_size": 1000
            }

            # 提取链路参数
            bit_error_rate = link_state.get("bit_error_rate", 1e-5)
            delay_ms = link_state.get("delay_ms", 100.0)
            transmission_rate_mbps = link_state.get("transmission_rate_mbps", 10.0)

            # 计算丢包率（用于tc命令）
            # from dtn_ion import calculate_packet_loss
            # loss_rate = calculate_packet_loss(bit_error_rate, bundle_size)
            loss_rate_percent = 0

            # 转换带宽为bit/s
            bandwidth_bps = int(transmission_rate_mbps * 1e6)

            print(f"\n[链路配置] 接收节点同步配置网络链路")
            print(f"  - Sequence: {sequence} (将用作EID)")
            print(f"  - 目标地址: {dest_addr}")
            print(f"  - 误码率: {bit_error_rate}")
            print(f"  - 延时: {delay_ms}ms")
            print(f"  - 传输速率: {transmission_rate_mbps}Mbps ({bandwidth_bps}bit/s)")
            print(f"  - 丢包率: {loss_rate_percent:.4f}%")
            print(f"  - Bundle大小: {bundle_size} bytes")

            # 调用configure_network配置网络
            if self.use_bp_ltp and self.bp_ltp_receiver:
                try:
                    # 根据sequence更新接收端的EID
                    self.bp_ltp_receiver.update_eid(sequence)

                    # 调用BP/LTP接收器的configure_network方法
                    self.bp_ltp_receiver.configure_network(
                        dest_addr=dest_addr,
                        bandwidth=bandwidth_bps,
                        tx_delay=int(delay_ms),
                        loss_rate=loss_rate_percent
                    )
                    print(f"[链路配置] 网络链路已配置完成")

                    # 启动BP/LTP接收监听
                    reception_started = self.start_bp_ltp_reception(data_size, bundle_size)
                    if reception_started:
                        print(f"[接收准备] BP/LTP接收监听已启动，准备接收数据")
                    else:
                        print(f"[警告] BP/LTP接收监听启动失败")

                except Exception as e:
                    print(f"[警告] BP/LTP接收器配置失败: {e}")
            else:
                print(f"[模拟模式] 在实际部署时，这里会调用tc命令配置网络")

            # 保存链路配置状态
            self.current_transmission["link_state"] = link_state
            self.current_transmission["bundle_size"] = bundle_size

            print(f"[链路配置] 链路配置处理完成")
            return True

        except Exception as e:
            print(f"[错误] 处理链路配置失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # 释放锁
            self.link_config_processing = False
            self.link_config_lock.release()
            print(f"[链路配置] 链路配置锁已释放")

    def handle_data_transmission(self, data: Dict[str, Any]) -> bool:
        """
        处理数据传输请求

        Args:
            data: 包含开始时间戳和数据大小的头部

        Returns:
            处理是否成功
        """
        try:
            start_timestamp = data.get("start_timestamp")
            data_size = data.get("data_size")

            self.current_transmission["start_timestamp"] = start_timestamp
            self.current_transmission["data_size"] = data_size

            start_time_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            print(f"[数据接收] 开始时间: {start_time_str}, 数据量: {data_size} bytes")

            return True

        except Exception as e:
            print(f"[错误] 处理数据传输失败: {e}")
            return False

    def handle_metadata(self, data: Dict[str, Any]) -> bool:
        """
        处理元数据请求（在接收完成后调用）

        Args:
            data: 包含传输元数据的数据

        Returns:
            处理是否成功
        """
        try:
            metadata = data.copy()

            data_size = metadata.get("data_size")
            start_timestamp = metadata.get("start_timestamp", 0.0)  # 从metadata中直接获取start_timestamp
            link_state = metadata.get("link_state", {})
            protocol_params = metadata.get("protocol_params", {})
            bundle_size = protocol_params.get("bundle_size", 1024)

            # 记录元数据（包括start_timestamp）
            self.current_transmission["data_size"] = data_size
            self.current_transmission["start_timestamp"] = start_timestamp
            self.current_transmission["link_state"] = link_state
            self.current_transmission["protocol_params"] = protocol_params

            print(f"[元数据接收] 已从metadata中获取start_timestamp: {start_timestamp}")

            # 如果启用了BP/LTP，等待接收完成
            if self.use_bp_ltp and self.bp_ltp_receiver:
                # 等待BP/LTP接收完成（最多等待6000秒）
                print(f"[等待接收] 等待BP/LTP接收完成...")
                if self.reception_event.wait(timeout=6000):
                    if self.reception_result.get("success"):
                        end_timestamp = self.reception_result.get("stop_time", time.time())
                        print(f"[接收完成] BP/LTP接收已完成，结束时间: {datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    else:
                        print(f"[警告] BP/LTP接收失败，使用当前时间")
                        end_timestamp = time.time()
                else:
                    print(f"[警告] BP/LTP接收超时，使用当前时间")
                    end_timestamp = time.time()
            else:
                # 模拟模式：直接使用当前时间
                print(f"[接收模式] 使用模拟模式（TCP接收已完成）")
                end_timestamp = time.time()

            # 使用从metadata中获取的start_timestamp
            # start_timestamp已经在上面从metadata中获取了

            # 验证时间戳有效性
            if start_timestamp <= 0 or start_timestamp > end_timestamp:
                print(f"[警告] 开始时间戳无效 (start={start_timestamp}, end={end_timestamp})")
                delivery_time_ms = 0.0
            else:
                delivery_time_ms = (end_timestamp - start_timestamp) * 1000

            start_time_str = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f') if start_timestamp > 0 else 'N/A'
            end_time_str = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            print(f"[时间计算] 开始: {start_timestamp}")
            print(f"[时间计算] 结束: {end_timestamp}")
            print(f"[传输指标] 业务交付时间: {delivery_time_ms:.3f}ms")

            # 生成训练记录
            self.logger.record_transmission(
                data_size=data_size,
                bit_error_rate=link_state.get("bit_error_rate", 0),
                delay_ms=link_state.get("delay_ms", 0),
                transmission_rate_mbps=link_state.get("transmission_rate_mbps", 0),
                bundle_size=protocol_params.get("bundle_size", 0),
                ltp_block_size=protocol_params.get("ltp_block_size", 0),
                ltp_segment_size=protocol_params.get("ltp_segment_size", 0),
                session_count=protocol_params.get("session_count", 0),
                delivery_time_ms=delivery_time_ms
            )

            return True

        except Exception as e:
            print(f"[错误] 处理元数据失败: {e}")
            return False

    def send_records_to_optimizer(self, records: List[Dict[str, Any]]) -> bool:
        """
        将记录发送到优化器(电脑C)

        Args:
            records: 要发送的记录列表

        Returns:
            发送是否成功
        """
        try:
            if not records:
                print("[警告] 没有要发送的记录")
                return True

            # 构造发送数据
            send_data = {
                "type": "training_records",
                "records": records,
                "count": len(records),
                "timestamp": time.time()
            }

            # 连接到优化器
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(25.0)
            sock.connect((self.optimizer_host, self.optimizer_port))

            # 发送数据
            send_json = json.dumps(send_data)
            message = send_json.encode('utf-8')

            # 先发送消息长度（4字节），再发送消息内容
            sock.sendall(struct.pack('!I', len(message)))
            sock.sendall(message)

            # 接收确认
            ack = sock.recv(1024)
            sock.close()

            print(f"[记录发送] 成功发送 {len(records)} 条记录到优化器")
            print(f"[确认信息] {ack.decode('utf-8')}")

            return True

        except Exception as e:
            print(f"[错误] 发送记录到优化器失败: {e}")
            return False

    def handle_client(self, client_socket: socket.socket, client_address: tuple):
        """
        处理单个客户端连接

        Args:
            client_socket: 客户端套接字
            client_address: 客户端地址
        """
        try:
            print(f"\n[新连接] 来自 {client_address}")

            # 接收消息长度
            length_data = client_socket.recv(4)
            if not length_data:
                return

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
            transmission_id = message.get("transmission_id")

            print(f"[消息类型] {message_type}, transmission_id={transmission_id}")

            # 消息去重：使用(transmission_id, message_type)组合作为唯一键
            # 同一轮传输的不同消息类型应该分别处理
            dedup_key = f"{transmission_id}:{message_type}" if transmission_id else None

            if dedup_key and dedup_key in self.processed_transmissions:
                print(f"[去重] 消息已处理过 (key={dedup_key})，直接返回ACK")
                client_socket.sendall(b"OK_ALREADY_PROCESSED")
                return

            if message_type == "link_config":
                # 链路配置请求：先快速发送ACK，避免发送端超时
                # 然后同步处理（单线程模式，确保顺序执行）
                print(f"[链路配置] 接收到链路配置请求，快速发送ACK...")
                client_socket.sendall(b"OK")
                print(f"[链路配置] ACK已发送，开始同步处理配置...")

                # 同步处理链路配置（不使用线程，确保单线程顺序执行）
                success = self.handle_link_config(message)

                if success:
                    print(f"[链路配置] 链路配置处理完成，等待数据传输")

                    # 记录已处理的消息（使用dedup_key）
                    if dedup_key:
                        self.processed_transmissions.add(dedup_key)
                        # 限制集合大小
                        if len(self.processed_transmissions) > self.max_processed_history:
                            oldest = list(self.processed_transmissions)[:len(self.processed_transmissions)//2]
                            for old_id in oldest:
                                self.processed_transmissions.discard(old_id)
                        print(f"[去重] 已记录key={dedup_key}（当前记录数: {len(self.processed_transmissions)}）")
                else:
                    print(f"[错误] 链路配置处理失败")

                # 处理完成，socket已经发送了ACK，直接返回
                return

            elif message_type == "data_transmission":
                success = self.handle_data_transmission(message)

                # 对于data_transmission，接收实际的数据负载（如果有）
                if "data_size" in message:
                    data_size = message["data_size"]
                    print(f"[数据接收] 准备接收 {data_size} 字节的数据负载...")
                    remaining_data = b''
                    while len(remaining_data) < data_size:
                        chunk = client_socket.recv(min(4096, data_size - len(remaining_data)))
                        if not chunk:
                            break
                        remaining_data += chunk
                    print(f"[数据接收] 已接收 {len(remaining_data)} 字节")

            elif message_type == "metadata":
                # metadata消息不包含额外的数据负载，只是元数据JSON
                success = self.handle_metadata(message)

            else:
                print(f"[警告] 未知的消息类型: {message_type}")
                success = False

            # 发送确认
            if success:
                # 记录已处理的消息（使用dedup_key）
                if dedup_key:
                    self.processed_transmissions.add(dedup_key)
                    # 限制集合大小，保留最近的N个
                    if len(self.processed_transmissions) > self.max_processed_history:
                        # 移除最早的（简单处理：清空一半）
                        oldest = list(self.processed_transmissions)[:len(self.processed_transmissions)//2]
                        for old_id in oldest:
                            self.processed_transmissions.discard(old_id)
                    print(f"[去重] 已记录key={dedup_key}（当前记录数: {len(self.processed_transmissions)}）")

                ack_message = "OK"
            else:
                ack_message = "FAILED"

            client_socket.sendall(ack_message.encode('utf-8'))

        except Exception as e:
            print(f"[错误] 处理客户端连接失败: {e}")

        finally:
            client_socket.close()

    def record_flusher_thread(self):
        """
        记录刷新线程 - 周期性地将记录发送到优化器
        """
        print("[记录刷新线程] 启动")

        while self.running:
            try:
                # 检查是否需要刷新
                if self.logger.should_flush():
                    records = self.logger.get_records_to_send()
                    if records:
                        print(f"\n[刷新记录] 正在发送 {len(records)} 条记录...")
                        self.send_records_to_optimizer(records)

                time.sleep(10)  # 检查间隔

            except Exception as e:
                print(f"[错误] 记录刷新线程异常: {e}")

    def run(self):
        """
        运行接收节点
        """
        print("="*60)
        print("接收节点B启动")
        print(f"监听端口: {self.listen_port}")
        print(f"优化器: {self.optimizer_host}:{self.optimizer_port}")
        print("="*60)

        # 启动记录刷新线程
        flusher_thread = threading.Thread(target=self.record_flusher_thread, daemon=True)
        flusher_thread.start()

        try:
            # 创建服务器套接字
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.listen_port))
            server_socket.listen(5)

            print(f"[监听] 在端口 {self.listen_port} 上监听来自节点A的连接...\n")

            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()

                    # 单线程模式：在主线程中直接处理，确保顺序执行
                    # 不创建新线程，处理完一个连接再accept下一个
                    print(f"[单线程] 在主线程中处理连接...")
                    self.handle_client(client_socket, client_address)
                    print(f"[单线程] 连接处理完成，继续监听下一个连接")

                except KeyboardInterrupt:
                    break

        except Exception as e:
            print(f"[错误] 服务器异常: {e}")

        finally:
            self.running = False
            server_socket.close()
            print("\n接收节点B停止")


def main():
    """主函数"""
    # 直接在代码中设置参数
    listen_port = 5001
    optimizer_host = '192.168.137.1'
    optimizer_port = 5003
    own_eid_number = 8
    use_bp_ltp = True  # 启用BP/LTP模式
    csv_file = "records_1.csv"
    sender_host = '192.168.137.194'  # 发送节点A的IP地址
    sender_notification_port = 5009  # 发送节点A的通知监听端口

    receiver = ReceiverNode(
        listen_port=listen_port,
        optimizer_host=optimizer_host,
        optimizer_port=optimizer_port,
        own_eid_number=own_eid_number,
        use_bp_ltp=use_bp_ltp,
        csv_file=csv_file,
        sender_host=sender_host,
        sender_notification_port=sender_notification_port
    )

    receiver.run()


if __name__ == "__main__":
    main()
