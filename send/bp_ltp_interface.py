#!/usr/bin/env python3
"""
BP/LTP协议栈接口封装
对dtn_ion.py中的API进行封装，便于在sender.py中调用
"""

import math
from typing import Dict
from dtn_ion import (
    configure_network,
    calculate_packet_loss,
    calculate_ltp_sessions,
    setup_ltp_span,
    send_bpdriver_command,
    setup_contact,
    extract_time_from_ionadmin
)


class BPLTPInterface:
    """BP/LTP协议栈接口类"""

    def __init__(self,
                 source_eid: str = "ipn:1.1",
                 destination_eid: int = 2,
                 dest_addr: str = "192.168.1.2",
                 dest_udp_addr: str = "192.168.1.2:1113",
                 bp_ttl: int = 3600):
        """
        初始化BP/LTP接口

        Args:
            source_eid: 源节点EID（例如 ipn:1.1）
            destination_eid: 目标节点EID（例如 2）
            dest_addr: 目标IP地址
            dest_udp_addr: 目标UDP地址（例如 192.168.1.2:1113）
            bp_ttl: Bundle生存时间（秒）
        """
        self.source_eid = source_eid
        self.destination_eid = destination_eid
        self.dest_addr = dest_addr
        self.dest_udp_addr = dest_udp_addr
        self.bp_ttl = bp_ttl
        self.destination_sequence = 2  # 默认为2，即 ipn:X.2

        # 当前协议栈参数
        self.current_bundle_size = 1024
        self.current_ltp_block = 512
        self.current_ltp_segment = 256
        self.current_ltp_sessions = 1
        self.ltp_aggregation_time = 5

        print(f"[BP/LTP接口] 初始化完成")
        print(f"  源节点: {self.source_eid}")
        print(f"  目标节点: ipn:{self.destination_eid}.{self.destination_sequence}")
        print(f"  目标地址: {self.dest_addr}")

    def update_destination_sequence(self, sequence: int):
        """
        根据sequence更新目标节点的EID后缀

        Args:
            sequence: CSV配置中的sequence字段，用作目标EID后缀
        """
        self.destination_sequence = sequence
        print(f"[BP/LTP接口] 目标EID后缀已更新")
        print(f"  新目标节点: ipn:{self.destination_eid}.{self.destination_sequence}")

    def configure_link_parameters(self,
                                  bit_error_rate: float,
                                  delay_ms: float,
                                  transmission_rate_mbps: float,
                                  data_size: int) -> Dict[str, float]:
        """
        根据链路状态配置网络参数

        Args:
            bit_error_rate: 误码率
            delay_ms: 延时（毫秒）
            transmission_rate_mbps: 传输速率（Mbps）
            data_size: 数据大小（bytes）

        Returns:
            配置信息字典
        """
        # 计算带宽（bit/s）
        bandwidth = int(transmission_rate_mbps * 1_000_000)

        # 计算丢包率（基于segment大小）
        loss_rate = calculate_packet_loss(bit_error_rate, self.current_ltp_segment)
        loss_rate_percent = loss_rate * 100

        print(f"\n[链路配置]")
        print(f"  误码率: {bit_error_rate}")
        print(f"  延时: {delay_ms}ms")
        print(f"  带宽: {bandwidth}bit/s ({transmission_rate_mbps}Mbps)")
        print(f"  丢包率: {loss_rate_percent:.4f}%")

        # 配置网络参数（使用tc命令）
        try:
            configure_network(self.dest_addr, bandwidth, int(delay_ms), loss_rate_percent)
            print(f"[链路配置] 成功配置网络参数")
        except Exception as e:
            print(f"[警告] 配置网络参数失败: {e}")

        return {
            "bandwidth": bandwidth,
            "delay_ms": delay_ms,
            "loss_rate": loss_rate_percent
        }

    def apply_protocol_parameters(self,
                                  bundle_size: int,
                                  ltp_block_size: int,
                                  ltp_segment_size: int,
                                  session_count: int = None,
                                  data_size: int = 10240,
                                  delay_ms: float = 100.0,
                                  transmission_rate_mbps: float = 10.0) -> bool:
        """
        应用协议栈参数到BP/LTP

        Args:
            bundle_size: Bundle大小（bytes）
            ltp_block_size: LTP Block大小（bytes）
            ltp_segment_size: LTP Segment大小（bytes）
            session_count: LTP会话数（如果为None则自动计算）
            data_size: 数据大小（bytes）
            delay_ms: 延时（毫秒）
            transmission_rate_mbps: 传输速率（Mbps）

        Returns:
            是否成功应用
        """
        self.current_bundle_size = bundle_size
        self.current_ltp_block = ltp_block_size
        self.current_ltp_segment = ltp_segment_size

        # 计算传输速率（Bytes/s）
        trans_rate = int(transmission_rate_mbps * 1_000_000 / 8)

        # 如果未指定会话数，则自动计算
        if session_count is None:
            ltp_sessions = calculate_ltp_sessions(
                delay=delay_ms,
                bundle_size=bundle_size,
                file_size=data_size,
                block_size=ltp_block_size,
                trans_rate=trans_rate
            )
        else:
            ltp_sessions = session_count

        self.current_ltp_sessions = ltp_sessions

        print(f"\n[协议参数配置]")
        print(f"  Bundle大小: {bundle_size} bytes")
        print(f"  LTP Block大小: {ltp_block_size} bytes")
        print(f"  LTP Segment大小: {ltp_segment_size} bytes")
        print(f"  LTP会话数: {ltp_sessions}")

        # 配置LTP span
        try:
            setup_ltp_span(
                destination_eid=self.destination_eid,
                ltp_sessions=ltp_sessions,
                ltp_segment=ltp_segment_size,
                ltp_block=ltp_block_size,
                ltp_aggregation_time=self.ltp_aggregation_time,
                dest_udp_addr=self.dest_udp_addr
            )
            print(f"[协议参数配置] 成功配置LTP span")
            return True
        except Exception as e:
            print(f"[错误] 配置LTP span失败: {e}")
            return False

    def setup_transmission_contact(self, transmission_rate_mbps: float) -> bool:
        """
        设置传输contact

        Args:
            transmission_rate_mbps: 传输速率（Mbps）

        Returns:
            是否成功设置
        """
        try:
            # 获取节点启动时间
            source_node = int(self.source_eid.split(':')[1].split('.')[0])
            node_start_time = extract_time_from_ionadmin(source_node, self.destination_eid)

            if not node_start_time:
                print(f"[警告] 无法获取节点启动时间，使用默认值")
                node_start_time = "+0"

            # 计算发送速率（Bytes/s）
            send_rate_B = int(transmission_rate_mbps * 1_000_000 / 8)

            # 设置contact
            setup_contact(
                node_start_time=node_start_time,
                own_eid=int(self.source_eid.split(':')[1].split('.')[0]),
                destination_eid=self.destination_eid,
                send_rate_B=send_rate_B
            )
            print(f"[Contact配置] 成功设置传输contact，速率: {send_rate_B} Bytes/s")
            return True
        except Exception as e:
            print(f"[错误] 设置contact失败: {e}")
            return False

    def transmit_data_via_bp_ltp(self,
                                 data_size: int,
                                 transmission_rate_mbps: float) -> float:
        """
        通过BP/LTP协议栈传输数据

        Args:
            data_size: 数据大小（bytes）
            transmission_rate_mbps: 传输速率（Mbps）

        Returns:
            发送时间戳（如果失败返回0）
        """
        try:
            # 计算传输周期数（基于bundle大小）
            nbr_of_cycles = math.ceil(data_size / self.current_bundle_size)

            # 计算发送速率（Bytes/s）
            send_rate_B = int(transmission_rate_mbps * 1_000_000 / 8)

            # 构造目标EID
            destination_eid = f"ipn:{self.destination_eid}.{self.destination_sequence}"

            print(f"\n[BP/LTP传输]")
            print(f"  数据大小: {data_size} bytes")
            print(f"  Bundle数量: {nbr_of_cycles}")
            print(f"  发送速率: {send_rate_B} Bytes/s")
            print(f"  源: {self.source_eid}")
            print(f"  目标: {destination_eid}")

            # 发送数据
            send_time = send_bpdriver_command(
                nbr_of_cycles=nbr_of_cycles,
                source=self.source_eid,
                destination=destination_eid,
                packet_size=self.current_bundle_size,
                BP_TTL=self.bp_ttl,
                send_rate_B=send_rate_B
            )

            print(f"[BP/LTP传输] 传输命令已发送，时间戳: {send_time}")
            return send_time

        except Exception as e:
            print(f"[错误] BP/LTP传输失败: {e}")
            return 0.0

    def get_current_parameters(self) -> Dict[str, int]:
        """
        获取当前协议栈参数

        Returns:
            当前参数字典
        """
        return {
            "bundle_size": self.current_bundle_size,
            "ltp_block_size": self.current_ltp_block,
            "ltp_segment_size": self.current_ltp_segment,
            "session_count": self.current_ltp_sessions
        }
