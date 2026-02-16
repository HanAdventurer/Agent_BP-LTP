#!/usr/bin/env python3
"""
BP/LTP接收器接口封装
对dtn_ion.py中的接收相关API进行封装，便于在receiver.py中调用
"""

import math
from typing import Tuple
from dtn_ion import run_bpcounter_and_monitor, configure_network


class BPLTPReceiverInterface:
    """BP/LTP接收器接口类"""

    def __init__(self, own_eid_number: int = 2):
        """
        初始化BP/LTP接收器接口

        Args:
            own_eid_number: 本节点EID数字（例如 2，对应 ipn:2.1）
        """
        self.own_eid_number = own_eid_number

        print(f"[BP/LTP接收器] 初始化完成")

    def update_eid(self, sequence: int):
        """
        根据sequence更新本节点的EID

        Args:
            sequence: CSV配置中的sequence字段，用作EID号
        """
        self.own_eid = f"ipn:{self.own_eid_number}.{sequence}"
        print(f"[BP/LTP接收器] EID已更新")
        print(f"  新EID: {self.own_eid}")

    def configure_network(self, dest_addr: str, bandwidth: int, tx_delay: int, loss_rate: float):
        """
        配置接收节点的网络链路参数，与发送节点保持一致

        Args:
            dest_addr: 目标地址（发送节点的IP）
            bandwidth: 带宽（bit/s）
            tx_delay: 传输延时（ms）
            loss_rate: 丢包率（%）
        """
        try:
            print(f"\n[BP/LTP接收器] 配置网络链路参数")
            print(f"  目标地址: {dest_addr}")
            print(f"  带宽: {bandwidth} bit/s")
            print(f"  延时: {tx_delay} ms")
            print(f"  丢包率: {loss_rate}%")

            # 调用dtn_ion.py中的configure_network函数
            configure_network(
                dest_addr=dest_addr,
                bandwidth=bandwidth,
                tx_delay=tx_delay,
                loss_rate=loss_rate
            )

            print(f"[BP/LTP接收器] 网络链路参数配置完成")

        except Exception as e:
            print(f"[错误] 配置网络失败: {e}")
            raise

    def calculate_bundle_count(self, data_size: int, bundle_size: int) -> int:
        """
        根据数据大小和bundle大小计算bundle数量

        Args:
            data_size: 总数据大小（bytes）
            bundle_size: 单个bundle大小（bytes）

        Returns:
            bundle数量（即bpcounter的max_count）
        """
        bundle_count = math.ceil(data_size / bundle_size)
        print(f"[Bundle计算] 数据大小: {data_size} bytes, "
              f"Bundle大小: {bundle_size} bytes, "
              f"Bundle数量: {bundle_count}")
        return bundle_count

    def monitor_reception(self, max_count: int) -> Tuple[str, float]:
        """
        监听BP/LTP数据接收过程

        Args:
            max_count: 要接收的bundle数量

        Returns:
            (bpcounter_report, stop_time): bpcounter输出报告和停止时间戳
        """
        try:
            print(f"\n[接收监听] 启动bpcounter监听")
            print(f"  本节点: {self.own_eid}")
            print(f"  预期接收Bundle数: {max_count}")

            # 调用bpcounter监听函数
            report, stop_time = run_bpcounter_and_monitor(
                source=self.own_eid,
                max_count=max_count
            )

            print(f"[接收完成] bpcounter已停止，停止时间戳: {stop_time}")
            print(f"[接收报告]:\n{report}")

            return report, stop_time

        except Exception as e:
            print(f"[错误] 监听接收失败: {e}")
            return "", 0.0

    def parse_bpcounter_report(self, report: str) -> dict:
        """
        解析bpcounter报告，提取关键性能指标

        Args:
            report: bpcounter的输出报告

        Returns:
            包含性能指标的字典
        """
        try:
            metrics = {
                "total_bundles_received": 0,
                "total_bytes_received": 0,
                "delivery_rate": 0.0,
                "report_raw": report
            }

            # 这里可以添加更复杂的报告解析逻辑
            # 根据实际的bpcounter输出格式进行调整
            lines = report.split('\n')
            for line in lines:
                line = line.strip()
                # 查找关键字段
                if 'delivered' in line.lower() or 'received' in line.lower():
                    print(f"[报告] {line}")

            print(f"[性能指标解析] 完成")
            return metrics

        except Exception as e:
            print(f"[错误] 解析bpcounter报告失败: {e}")
            return {}