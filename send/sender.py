#!/usr/bin/env python3
"""
发送节点A (Sender Node A)
功能：
1. 产生业务请求
2. 收集链路状态和协议参数
3. 向电脑C请求优化参数
4. 应用优化参数并向节点B传输数据
5. 发送传输元数据给节点B
"""

import socket
import json
import time
import struct
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入BP/LTP接口
try:
    from bp_ltp_interface import BPLTPInterface
    BP_LTP_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入BP/LTP接口，将使用模拟模式")
    BP_LTP_AVAILABLE = False


class SenderNode:
    def __init__(self,
                 receiver_host: str = "192.168.1.2",  # 节点B的IP
                 receiver_port: int = 5001,
                 optimizer_host: str = "192.168.1.3",  # 电脑C的IP
                 optimizer_port: int = 5002,
                 config_file: str = "network_config.csv",
                 source_eid: str = "ipn:1.1",
                 destination_eid: str = "ipn:2.1",
                 dest_udp_addr: str = "192.168.1.2:1113",
                 own_host: str = "192.168.1.1",  # 本节点（发送节点A）的IP地址
                 use_bp_ltp: bool = True):
        """
        初始化发送节点

        Args:
            receiver_host: 接收节点B的IP地址
            receiver_port: 接收节点B的端口
            optimizer_host: 优化器(电脑C)的IP地址
            optimizer_port: 优化器的端口
            config_file: 网络配置CSV文件路径
            source_eid: 源节点EID（例如 ipn:1.1）
            destination_eid: 目标节点EID数字（例如 2）
            dest_udp_addr: 目标UDP地址（例如 192.168.1.2:1113）
            own_host: 本节点（发送节点A）的IP地址
            use_bp_ltp: 是否使用BP/LTP协议栈
        """
        self.receiver_host = receiver_host
        self.receiver_port = receiver_port
        self.optimizer_host = optimizer_host
        self.optimizer_port = optimizer_port
        self.config_file = config_file
        self.own_host = own_host



        # 当前协议栈参数（默认值）
        self.protocol_params = {
            "bundle_size": 1000,        # bundle大小 (bytes)
            "ltp_block_size": 160000,      # LTP Block大小 (bytes)
            "ltp_segment_size": 1400,    # LTP segment大小 (bytes)
            "session_count": 10          # 会话数量
        }

        # BP/LTP接收完成通知机制（单线程模式）
        self.notification_listener_port = 5009  # 用于接收接收端的完成通知

        # 消息去重：为每轮传输生成唯一ID
        self.current_transmission_id = None

        # 从CSV加载配置
        self.config_data = self.load_config_from_csv()
        self.config_index = 0  # 当前配置索引

        # 初始化BP/LTP接口
        self.use_bp_ltp = use_bp_ltp and BP_LTP_AVAILABLE
        self.bp_ltp_interface = None

        if self.use_bp_ltp:
            try:
                self.bp_ltp_interface = BPLTPInterface(
                    source_eid=source_eid,
                    destination_eid=destination_eid,
                    dest_addr=receiver_host,
                    dest_udp_addr=dest_udp_addr
                )
                print(f"[初始化] BP/LTP接口已启用")
            except Exception as e:
                print(f"[警告] 初始化BP/LTP接口失败: {e}")
                self.use_bp_ltp = False

    def wait_for_reception_completion(self, timeout: float = 300) -> bool:
        """
        等待接收端的完成通知（单线程模式，主线程直接阻塞监听）

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否收到完成通知
        """
        print(f"[等待接收] 主线程开始监听端口 {self.notification_listener_port}（超时{timeout}秒）")

        try:
            # 创建监听socket
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(('0.0.0.0', self.notification_listener_port))
            server_sock.listen(1)
            server_sock.settimeout(timeout)  # 设置超时

            print(f"[等待接收] 正在等待接收端的完成通知...")

            try:
                # 阻塞等待接收端连接
                client_sock, client_addr = server_sock.accept()
                print(f"[等待接收] 收到来自 {client_addr} 的连接")

                client_sock.settimeout(30.0)  # 客户端socket超时

                try:
                    # 接收消息长度
                    length_data = client_sock.recv(4)
                    if not length_data:
                        print(f"[警告] 未接收到消息长度")
                        client_sock.close()
                        server_sock.close()
                        return False

                    message_length = struct.unpack('!I', length_data)[0]
                    print(f"[等待接收] 消息长度: {message_length} 字节")

                    # 接收完整消息
                    message_data = b''
                    while len(message_data) < message_length:
                        chunk = client_sock.recv(min(4096, message_length - len(message_data)))
                        if not chunk:
                            print(f"[警告] 接收消息数据时连接断开")
                            break
                        message_data += chunk

                    if len(message_data) < message_length:
                        print(f"[警告] 消息接收不完整: {len(message_data)}/{message_length}")
                        client_sock.close()
                        server_sock.close()
                        return False

                    # 解析消息
                    message = json.loads(message_data.decode('utf-8'))
                    msg_type = message.get("type")
                    print(f"[等待接收] 消息类型: {msg_type}")

                    if msg_type == "reception_complete":
                        print(f"[等待接收] ✅ 接收到接收完成通知")

                        # 发送确认（接收端会阻塞等待这个确认）
                        client_sock.sendall(b"ACK")
                        print(f"[等待接收] ACK已发送给接收端")

                        client_sock.close()
                        server_sock.close()
                        return True
                    else:
                        print(f"[等待接收] ⚠️  未知消息类型: {msg_type}")
                        client_sock.close()
                        server_sock.close()
                        return False

                except Exception as inner_e:
                    print(f"[错误] 处理消息时出错: {inner_e}")
                    import traceback
                    traceback.print_exc()
                    client_sock.close()
                    server_sock.close()
                    return False

            except socket.timeout:
                print(f"[超时] 等待接收完成通知超时（{timeout}秒）")
                server_sock.close()
                return False

        except Exception as e:
            print(f"[错误] 监听端口异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def send_start_timestamp_to_receiver(self, start_timestamp: float, data_size: int, max_attempts: int = 30) -> bool:
        """
        发送开始时间戳到接收端（使用重试+指数退避，最多尝试max_attempts次）

        Args:
            start_timestamp: BP/LTP传输开始时间戳
            data_size: 数据大小
            max_attempts: 最大重试次数（默认30次，约30分钟）

        Returns:
            是否发送成功
        """
        import random

        attempt = 0
        backoff = 1.0  # 初始退避时间（秒）
        max_backoff = 60.0  # 最大退避时间（秒）

        print(f"[时间同步] 开始发送BP/LTP开始时间戳到接收端（最多尝试{max_attempts}次）...")

        while attempt < max_attempts:
            attempt += 1

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30.0)
                sock.connect((self.receiver_host, self.receiver_port))

                # 准备消息（BP/LTP模式下只传递时间戳，不发送数据负载）
                header = {
                    "type": "start_timestamp",  # 使用独立的消息类型，避免与data_transmission混淆
                    "transmission_id": self.current_transmission_id,  # 消息去重ID
                    "start_timestamp": start_timestamp,
                    "data_size": data_size
                }
                header_json = json.dumps(header).encode('utf-8')
                message_len = struct.pack('!I', len(header_json))

                # 发送消息
                sock.sendall(message_len)
                sock.sendall(header_json)

                # 等待确认
                ack = sock.recv(1024)
                sock.close()

                if ack:
                    ack_text = ack.decode('utf-8', errors='ignore')
                    if "ALREADY_PROCESSED" in ack_text:
                        print(f"[时间同步] 接收端已处理过此消息 (transmission_id={self.current_transmission_id})，无需重复发送")
                        return True
                    else:
                        print(f"[时间同步] 成功发送开始时间戳到接收端（第{attempt}次尝试），确认: {ack_text}")
                        return True

            except Exception as e:
                print(f"[警告] 第{attempt}次发送开始时间戳失败: {e}")

                # 如果已达最大尝试次数，返回失败
                if attempt >= max_attempts:
                    print(f"[错误] 已达最大重试次数{max_attempts}，放弃发送开始时间戳")
                    return False

                # 计算下次重试的等待时间（指数退避 + 随机抖动）
                sleep_time = backoff + random.uniform(0, 0.5)
                print(f"[重试] 在 {sleep_time:.1f}s 后进行第{attempt + 1}次尝试")
                time.sleep(sleep_time)

                # 更新退避时间（指数增长，但不超过最大值）
                backoff = min(max_backoff, backoff * 2)

                # 关闭socket
                try:
                    sock.close()
                except:
                    pass

        return False

    def load_config_from_csv(self) -> list:
        """
        从CSV文件中加载配置数据

        Returns:
            配置数据列表，每个元素为一个字典
        """
        config_data = []

        try:
            # 检查文件是否存在
            if not os.path.exists(self.config_file):
                print(f"[警告] 配置文件 {self.config_file} 不存在，使用默认参数")
                return config_data

            # 读取CSV文件
            with open(self.config_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader):
                    try:
                        config_entry = {
                            "sequence": int(row.get("sequence", row_idx + 1)),
                            "data_size": int(row.get("data_size_bytes", 10240)),
                            "bit_error_rate": float(row.get("bit_error_rate", 1e-5)),
                            "delay_ms": float(row.get("delay_ms", 100.0)),
                            "transmission_rate_mbps": float(row.get("transmission_rate_mbps", 10.0)),
                            "description": row.get("description", "")
                        }
                        config_data.append(config_entry)
                    except ValueError as e:
                        print(f"[警告] 第 {row_idx + 2} 行数据解析失败: {e}")
                        continue

            print(f"[配置加载] 成功从 {self.config_file} 加载 {len(config_data)} 条配置")
            return config_data

        except Exception as e:
            print(f"[错误] 读取配置文件失败: {e}")
            return config_data

    def get_link_state(self) -> Dict[str, float]:
        """
        获取当前链路状态
        如果配置文件存在，从CSV中读取；否则使用默认值

        Returns:
            包含误码率、延时、传输速率和sequence的字典
        """
        if self.config_data and len(self.config_data) > 0:
            # 从CSV配置中循环读取
            current_config = self.config_data[self.config_index % len(self.config_data)]

            link_state = {
                "bit_error_rate": current_config["bit_error_rate"],
                "delay_ms": current_config["delay_ms"],
                "transmission_rate_mbps": current_config["transmission_rate_mbps"],
                "sequence": (current_config["sequence"] % 27) + 2  # 添加sequence字段，用于接收端EID配置
            }

            print(f"[CSV配置 {current_config['sequence']}] {current_config.get('description', '')}")
            return link_state
        else:
            # 使用默认值
            # TODO: 在实际部署时，也可以从真实的链路监测接口获取这些值
            link_state = {
                "bit_error_rate": 1e-5,      # 误码率
                "delay_ms": 100.0,            # 延时 (毫秒)
                "transmission_rate_mbps": 10.0,  # 传输速率 (Mbps)
                "sequence": 2  # 默认sequence为1
            }
            return link_state

    def generate_business_request(self) -> int:
        """
        产生业务请求，返回待发送的数据量大小
        如果配置文件存在，从CSV中读取；否则使用默认值

        Returns:
            数据量大小 (bytes)
        """
        if self.config_data and len(self.config_data) > 0:
            # 从CSV配置中循环读取
            current_config = self.config_data[self.config_index % len(self.config_data)]
            data_size = current_config["data_size"]
            print(f"[业务请求] 待发送数据量: {data_size} bytes (从CSV配置读取)")
            return data_size
        else:
            # 使用默认值
            # TODO: 在实际应用中，这应该来自真实的业务层
            data_size = 10240  # 10KB示例
            print(f"[业务请求] 待发送数据量: {data_size} bytes (默认值)")
            return data_size

    def request_optimized_params(self, data_size: int, link_state: Dict[str, float]) -> Dict[str, int]:
        """
        向优化器(电脑C)请求优化后的协议参数

        Args:
            data_size: 待发送数据量大小
            link_state: 链路状态

        Returns:
            优化后的协议参数
        """
        try:
            # 构造请求数据
            request_data = {
                "data_size": data_size,
                "link_state": link_state,
                "current_params": self.protocol_params,
                "timestamp": time.time()
            }

            # 连接到优化器
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.optimizer_host, self.optimizer_port))

            # 发送请求
            request_json = json.dumps(request_data)
            message = request_json.encode('utf-8')

            # 先发送消息长度（4字节），再发送消息内容
            sock.sendall(struct.pack('!I', len(message)))
            sock.sendall(message)

            print(f"[请求优化] 已发送请求到优化器 {self.optimizer_host}:{self.optimizer_port}")

            # 接收优化后的参数
            length_data = sock.recv(4)
            if not length_data:
                raise Exception("未收到优化器响应")

            message_length = struct.unpack('!I', length_data)[0]
            response_data = b''
            while len(response_data) < message_length:
                chunk = sock.recv(min(4096, message_length - len(response_data)))
                if not chunk:
                    break
                response_data += chunk

            sock.close()

            # 解析响应
            response = json.loads(response_data.decode('utf-8'))
            optimized_params = response.get("optimized_params", self.protocol_params)

            print(f"[收到优化参数] {optimized_params}")
            return optimized_params

        except Exception as e:
            print(f"[错误] 请求优化参数失败: {e}")
            print("[警告] 使用当前协议参数继续传输")
            return self.protocol_params

    def apply_protocol_params(self, params: Dict[str, int], link_state: Dict[str, float] = None, data_size: int = 10240):
        """
        应用协议栈参数

        Args:
            params: 要应用的协议参数
            link_state: 链路状态（用于BP/LTP配置）
            data_size: 数据大小（用于BP/LTP配置）
        """
        self.protocol_params.update(params)
        print(f"[参数应用] 已更新协议栈参数: {self.protocol_params}")

        # 如果启用了BP/LTP接口，则配置协议栈
        if self.use_bp_ltp and self.bp_ltp_interface:
            try:
                # 1. 配置链路参数（如果提供了link_state）
                if link_state:
                    self.bp_ltp_interface.configure_link_parameters(
                        bit_error_rate=link_state.get("bit_error_rate", 1e-5),
                        delay_ms=link_state.get("delay_ms", 100.0),
                        transmission_rate_mbps=link_state.get("transmission_rate_mbps", 10.0),
                        data_size=data_size
                    )

                # 2. 应用协议栈参数
                success = self.bp_ltp_interface.apply_protocol_parameters(
                    bundle_size=params.get("bundle_size", 10000),
                    ltp_block_size=params.get("ltp_block_size", 160000),
                    ltp_segment_size=params.get("ltp_segment_size", 1400),
                    session_count=params.get("session_count"),
                    data_size=data_size,
                    delay_ms=link_state.get("delay_ms", 100.0) if link_state else 100.0,
                    transmission_rate_mbps=link_state.get("transmission_rate_mbps", 10.0) if link_state else 10.0
                )

                if success:
                    print(f"[BP/LTP配置] 成功应用协议栈参数到ION")
                else:
                    print(f"[警告] BP/LTP配置失败，但会继续传输")

            except Exception as e:
                print(f"[警告] BP/LTP配置异常: {e}")
        else:
            # 模拟模式（原有的TODO注释）
            print(f"[模拟模式] 在实际部署时，这里会调用BP/LTP协议栈的配置接口")

    def send_link_config_to_receiver(self, data_size: int, link_state: Dict[str, float], max_attempts: int = 10) -> bool:
        """
        在传输数据之前，向接收节点B发送链路配置信息，让接收节点同步配置网络
        使用重试机制确保接收端一定能收到

        Args:
            data_size: 待发送数据量大小
            link_state: 链路状态
            max_attempts: 最大重试次数（默认10次）

        Returns:
            发送是否成功
        """
        import random

        # 从link_state中提取sequence字段（用于接收端EID配置）
        sequence = link_state.get("sequence", 1)

        # 构造链路配置信息
        link_config = {
            "type": "link_config",
            "transmission_id": self.current_transmission_id,  # 消息去重ID
            "data_size": data_size,
            "bundle_size": self.protocol_params['bundle_size'],
            "link_state": link_state,
            "dest_addr": self.own_host,  # 发送节点的IP地址
            "sequence": sequence,  # 添加sequence字段，用于接收端EID配置
            "timestamp": time.time()
        }

        config_json = json.dumps(link_config).encode('utf-8')
        message_len = struct.pack('!I', len(config_json))

        attempt = 0
        backoff = 1.0
        max_backoff = 10.0

        print(f"[链路配置] 准备发送链路配置到节点B (sequence={sequence}, 最多尝试{max_attempts}次)...")

        while attempt < max_attempts:
            attempt += 1

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((self.receiver_host, self.receiver_port))

                # 发送链路配置
                sock.sendall(message_len)
                sock.sendall(config_json)

                # 等待接收节点确认
                ack = sock.recv(1024)
                sock.close()

                if ack:
                    ack_text = ack.decode('utf-8', errors='ignore')
                    if "ALREADY_PROCESSED" in ack_text:
                        print(f"[链路配置] 接收端已处理过此消息 (transmission_id={self.current_transmission_id})，无需重复发送")
                        return True
                    else:
                        print(f"[链路配置] 已发送链路配置到节点B (第{attempt}次尝试)，接收节点确认: {ack_text}")
                        return True

            except Exception as e:
                print(f"[警告] 第{attempt}次发送链路配置失败: {e}")

                # 如果已达最大尝试次数，返回失败
                if attempt >= max_attempts:
                    print(f"[错误] 已达最大重试次数{max_attempts}，链路配置发送失败")
                    return False

                # 计算下次重试的等待时间（指数退避 + 随机抖动）
                sleep_time = backoff + random.uniform(0, 0.3)
                print(f"[重试] 在 {sleep_time:.1f}s 后进行第{attempt + 1}次尝试")
                time.sleep(sleep_time)

                # 更新退避时间
                backoff = min(max_backoff, backoff * 2)

                # 关闭socket
                try:
                    sock.close()
                except:
                    pass

        return False

    def transmit_data(self, data_size: int, link_state: Dict[str, float] = None) -> tuple:
        """
        向接收节点B传输数据

        Args:
            data_size: 数据量大小
            link_state: 链路状态

        Returns:
            (start_timestamp, success)
        """
        try:
            # 步骤1：在传输数据之前，发送链路配置信息到接收节点
            if link_state:
                config_sent = self.send_link_config_to_receiver(data_size, link_state)
                if not config_sent:
                    print(f"[警告] 链路配置发送失败，但会继续传输")
                else:
                    print(f"[链路配置] 接收节点已确认配置完成，准备开始传输")

            # 记录开始时间戳（用于TCP模拟模式）
            start_timestamp = time.time()

            # 如果启用了BP/LTP接口，使用BP/LTP进行传输
            if self.use_bp_ltp and self.bp_ltp_interface:
                try:
                    transmission_rate = link_state.get("transmission_rate_mbps", 10.0) if link_state else 10.0
                    sequence = link_state.get("sequence", 1) if link_state else 1

                    print(f"[传输参数] Bundle大小: {self.protocol_params['bundle_size']}, "
                        f"LTP Block大小: {self.protocol_params['ltp_block_size']}, "
                        f"LTP Segment大小: {self.protocol_params['ltp_segment_size']}, "
                        f"会话数: {self.protocol_params['session_count']}")

                    # 步骤1：根据sequence更新目标EID
                    self.bp_ltp_interface.update_destination_sequence(sequence)

                    # 步骤2：设置transmission contact
                    self.bp_ltp_interface.setup_transmission_contact(transmission_rate)
                    time.sleep(3)
                    # 步骤3：通过BP/LTP发送数据，获取真正的发送时间戳
                    bp_send_time = self.bp_ltp_interface.transmit_data_via_bp_ltp(
                        data_size=data_size,
                        transmission_rate_mbps=transmission_rate
                    )

                    if bp_send_time > 0:
                        print(f"[BP/LTP传输] 成功通过BP/LTP协议栈发送数据")
                        bp_send_time_str = datetime.fromtimestamp(bp_send_time).strftime('%Y-%m-%d %H:%M:%S.%f')
                        print(f"[开始时间] BP/LTP发送时间戳: {bp_send_time_str}")
                        # 使用BP/LTP的实际发送时间戳
                        start_timestamp = bp_send_time

                        # 步骤4：等待接收端的完成通知（主线程阻塞）
                        print(f"[等待接收] 等待接收端BP/LTP接收完成通知...")
                        completion_received = self.wait_for_reception_completion(timeout=3000)

                        if completion_received:
                            print(f"[接收完成] 收到接收端完成通知")

                            # 步骤5：接收完成后，发送真正的bp_send_time给接收端（确保发送成功）
                            success = self.send_start_timestamp_to_receiver(
                                start_timestamp=start_timestamp,
                                data_size=data_size,
                                max_attempts=60  # 最多尝试60次
                            )

                            if success:
                                print(f"[传输流程] BP/LTP传输流程完整结束")
                                return start_timestamp, True
                            else:
                                print(f"[错误] 发送开始时间戳失败，BP/LTP传输流程不完整")
                                return start_timestamp, False
                        else:
                            print(f"[错误] 等待接收完成超时，BP/LTP传输流程不完整")
                            return start_timestamp, False
                    else:
                        print(f"[警告] BP/LTP传输失败，回退到模拟模式")

                except Exception as e:
                    print(f"[警告] BP/LTP传输异常: {e}，回退到模拟模式")

            # 模拟模式：通过TCP发送数据到接收节点
            print(f"[传输模式] 使用TCP模拟模式发送数据")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.connect((self.receiver_host, self.receiver_port))

            # 构造传输头部（包含时间戳和数据大小）
            header = {
                "start_timestamp": start_timestamp,
                "data_size": data_size,
                "type": "data_transmission"
            }
            header_json = json.dumps(header).encode('utf-8')

            # 发送头部
            sock.sendall(struct.pack('!I', len(header_json)))
            sock.sendall(header_json)

            # 模拟发送数据
            dummy_data = b'X' * data_size
            sock.sendall(dummy_data)

            # 等待接收确认
            ack = sock.recv(1024)
            sock.close()

            print(f"[传输完成] 接收节点确认: {ack.decode('utf-8')}")
            return start_timestamp, True

        except Exception as e:
            print(f"[错误] 数据传输失败: {e}")
            return start_timestamp, False

    def send_metadata(self, data_size: int, link_state: Dict[str, float], max_attempts: int = 10) -> bool:
        """
        传输结束后，向接收节点B发送传输元数据
        使用重试机制确保元数据一定能发送成功

        Args:
            data_size: 数据量大小
            link_state: 链路状态
            max_attempts: 最大重试次数（默认10次）

        Returns:
            发送是否成功
        """
        import random

        # 构造元数据
        metadata = {
            "type": "metadata",
            "transmission_id": self.current_transmission_id,  # 消息去重ID
            "data_size": data_size,
            "link_state": link_state,
            "protocol_params": self.protocol_params,
            "timestamp": time.time()
        }

        metadata_json = json.dumps(metadata).encode('utf-8')
        message_len = struct.pack('!I', len(metadata_json))

        attempt = 0
        backoff = 1.0
        max_backoff = 10.0

        print(f"[元数据发送] 准备发送传输元数据（最多尝试{max_attempts}次）...")

        while attempt < max_attempts:
            attempt += 1

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((self.receiver_host, self.receiver_port))

                # 发送元数据
                sock.sendall(message_len)
                sock.sendall(metadata_json)

                # 等待接收节点确认
                ack = sock.recv(1024)
                sock.close()

                if ack:
                    ack_text = ack.decode('utf-8', errors='ignore')
                    if "ALREADY_PROCESSED" in ack_text:
                        print(f"[元数据发送] 接收端已处理过此消息 (transmission_id={self.current_transmission_id})，无需重复发送")
                        return True
                    else:
                        print(f"[元数据发送] 已发送传输元数据到节点B（第{attempt}次尝试），确认: {ack_text}")
                        return True

            except Exception as e:
                print(f"[警告] 第{attempt}次发送元数据失败: {e}")

                # 如果已达最大尝试次数，返回失败
                if attempt >= max_attempts:
                    print(f"[错误] 已达最大重试次数{max_attempts}，元数据发送失败")
                    return False

                # 计算下次重试的等待时间
                sleep_time = backoff + random.uniform(0, 0.3)
                print(f"[重试] 在 {sleep_time:.1f}s 后进行第{attempt + 1}次尝试")
                time.sleep(sleep_time)

                # 更新退避时间
                backoff = min(max_backoff, backoff * 2)

                # 关闭socket
                try:
                    sock.close()
                except:
                    pass

        return False

    def run_transmission_cycle(self):
        """
        执行一次完整的传输周期
        """
        print("\n" + "="*60)
        print(f"开始新的传输周期 (配置索引: {self.config_index})")
        print("="*60)

        try:
            # 0. 生成本轮传输的唯一ID（用于消息去重）
            self.current_transmission_id = f"{int(time.time() * 1000)}_{self.config_index}"
            print(f"[传输ID] {self.current_transmission_id}")

            # 1. 产生业务请求
            data_size = self.generate_business_request()

            # 2. 获取链路状态
            link_state = self.get_link_state()
            print(f"[链路状态] 误码率: {link_state['bit_error_rate']}, "
                  f"延时: {link_state['delay_ms']}ms, "
                  f"速率: {link_state['transmission_rate_mbps']}Mbps")

            # 3. 请求优化参数
            optimized_params = self.request_optimized_params(data_size, link_state)

            # 4. 应用优化参数（传递链路状态和数据大小）
            self.apply_protocol_params(optimized_params, link_state=link_state, data_size=data_size)

            # 5. 传输数据（传递链路状态）
            start_timestamp, success = self.transmit_data(data_size, link_state=link_state)

            if success:
                # 6. 发送元数据（确保发送成功）
                metadata_sent = self.send_metadata(data_size, link_state, max_attempts=10)

                if metadata_sent:
                    print("\n[周期完成] 传输周期成功完成\n")
                else:
                    print("\n[周期失败] 元数据发送失败，传输周期不完整\n")
            else:
                print("\n[周期失败] 数据传输失败\n")

        except Exception as e:
            print(f"\n[周期异常] 传输周期发生异常: {e}\n")
            import traceback
            traceback.print_exc()

        finally:
            # 更新配置索引（循环使用）
            if self.config_data:
                self.config_index = (self.config_index + 1) % len(self.config_data)

            print(f"[周期结束] 本轮传输流程完全结束，准备下一轮\n")

    def run(self, interval: int = 60):
        """
        持续运行发送节点（单线程模式）

        Args:
            interval: 传输周期间隔（秒）
        """
        print("="*60)
        print("发送节点A启动（单线程模式）")
        print(f"接收节点B: {self.receiver_host}:{self.receiver_port}")
        print(f"优化器C: {self.optimizer_host}:{self.optimizer_port}")
        print("="*60)

        try:
            while True:
                self.run_transmission_cycle()
                print(f"等待 {interval} 秒后进行下一次传输...")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n发送节点A停止")


def main():
    """主函数"""
    # 直接在代码中设置参数
    receiver_host = '192.168.137.164'  # 接收节点B的IP地址
    receiver_port = 5001
    optimizer_host = '192.168.137.1'  # 优化器(电脑C)的IP地址
    optimizer_port = 5002
    own_host = '192.168.137.194'  # 本节点（发送节点A）的IP地址
    config_file = 'agent_sender/network_config.csv'
    interval = 20  # 传输周期间隔（秒）

    # BP/LTP相关参数
    source_eid = 'ipn:9.2'
    destination_eid = 8
    dest_udp_addr = '192.168.137.164:1113'
    use_bp_ltp = True  # 启用BP/LTP协议栈

    sender = SenderNode(
        receiver_host=receiver_host,
        receiver_port=receiver_port,
        optimizer_host=optimizer_host,
        optimizer_port=optimizer_port,
        config_file=config_file,
        own_host=own_host,
        source_eid=source_eid,
        destination_eid=destination_eid,
        dest_udp_addr=dest_udp_addr,
        use_bp_ltp=use_bp_ltp
    )

    sender.run(interval=interval)


if __name__ == "__main__":
    main()
