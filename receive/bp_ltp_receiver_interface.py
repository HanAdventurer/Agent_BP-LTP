#!/usr/bin/env python3
"""
BP/LTP接收器接口封装
对dtn_ion.py中的接收相关API进行封装，便于在receiver.py中调用
"""

import math
from typing import Tuple
from dtn_ion import run_bpcounter_and_monitor, configure_network
import subprocess
import time

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
    def run_bpcounter_with_timeout(self, timeout: int = 5) -> int:
        """
        在正式接收数据前，运行bpcounter清理遗留数据

        作用：清理上一轮传输可能遗留在网络中的数据包，防止干扰本轮接收

        Args:
            timeout: 清理持续时间（秒），默认5秒

        Returns:
            清理的遗留数据包数量（bundle count）
        """
        command = f"sudo bpcounter {self.own_eid}"
        print(f"\n[遗留数据清理] 启动bpcounter清理遗留数据（持续{timeout}秒）")
        print(f"  命令: {command}")

        try:
            process = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            start_time = time.time()
            output_lines = []
            residual_count = 0

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                line = line.strip()
                if line:
                    output_lines.append(line)
                    # 统计接收到的bundle（遗留数据）
                    if "bundles" in line.lower() and "received" in line.lower():
                        print(f"  [遗留数据] {line}")

                # 超时后终止
                if time.time() - start_time > timeout:
                    print(f"[遗留数据清理] {timeout}秒超时，终止清理进程")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break

            # 解析输出，提取遗留数据统计
            full_output = "\n".join(output_lines)
            import re
            # 尝试从报告中提取bundle数量
            match = re.search(r'(\d+)\s+bundles?\s+received', full_output, re.IGNORECASE)
            if match:
                residual_count = int(match.group(1))

            if residual_count > 0:
                print(f"[遗留数据清理] ✅ 清理完成，共清理 {residual_count} 个遗留bundle")
            else:
                print(f"[遗留数据清理] ✅ 清理完成，无遗留数据")

            return residual_count

        except Exception as e:
            print(f"[错误] 遗留数据清理失败: {e}")
            return 0
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
            # print(f"[接收报告]:\n{report}")

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
            import re

            metrics = {
                "total_bundles_received": 0,
                "total_bytes_received": 0,
                "delivery_rate": 0.0,
                "report_raw": report
            }

            # 解析bpcounter报告
            # 查找"Total bytes:"字段
            lines = report.split('\n')
            for line in lines:
                line = line.strip()

                # 匹配 "Total bytes: XXXXX" 格式
                match = re.search(r'Total\s+bytes:\s*(\d+)', line, re.IGNORECASE)
                if match:
                    total_bytes = int(match.group(1))
                    metrics["total_bytes_received"] = total_bytes
                    print(f"[报告解析] Total bytes: {total_bytes}")

                # 可以继续解析其他字段（如果需要）
                # 例如：Total bundles, Delivery rate等
                if 'bundles' in line.lower() and 'total' in line.lower():
                    bundle_match = re.search(r'(\d+)', line)
                    if bundle_match:
                        metrics["total_bundles_received"] = int(bundle_match.group(1))
                        print(f"[报告解析] Total bundles: {metrics['total_bundles_received']}")

            print(f"[性能指标解析] 完成 - 接收字节数: {metrics['total_bytes_received']}")
            return metrics

        except Exception as e:
            print(f"[错误] 解析bpcounter报告失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "total_bundles_received": 0,
                "total_bytes_received": 0,
                "delivery_rate": 0.0,
                "report_raw": report
            }