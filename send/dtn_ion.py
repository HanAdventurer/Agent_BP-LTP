import subprocess
import time
import os
import re
import socket
import csv
from datetime import datetime
import math
import threading
from threading import Event
from queue import Queue, Empty
import random
import math
from typing import Union

def run_ion_commands(base_path, subdirectory):
    """执行 bench 命令序列"""
    full_subdirectory_path = os.path.join(base_path, subdirectory)
    print(f"正在执行: cd {full_subdirectory_path} && ./ionstop")
    os.chdir(full_subdirectory_path)
    execute_command("./ionstop")
    time.sleep(20)
    print(f"正在执行: cd {base_path} && sudo ./cleanup")
    os.chdir(base_path)
    execute_command("./cleanup")
    time.sleep(1)
    print("正在执行: ionstop")
    execute_command("ionstop")
    time.sleep(20)
    print(f"正在执行: cd {full_subdirectory_path} && ./ionstart")
    os.chdir(full_subdirectory_path)
    execute_command("./ionstart")
    time.sleep(30)
    print("所有命令执行完成！")

def calculate_packet_loss(bit_error_rate, packet_size_bytes):
    ber = bit_error_rate
    packet_size_bits = packet_size_bytes * 8
    return 1 - (1 - ber) ** packet_size_bits


def execute_command(command):
    """执行命令"""
    try:
        os.system(command)
    except Exception as e:
        print(f"An error occurred while executing the command: {e}")


def configure_network(dest_addr, bandwidth, tx_delay, loss_rate):
    # bandwidth = "50"
    """配置网络参数"""
    execute_command("sudo tc qdisc del dev eth0 root")
    execute_command("sudo tc qdisc add dev eth0 root handle 1: htb")
    execute_command(f"sudo tc class add dev eth0 parent 1: classid 1:10 htb rate {bandwidth}kbit")
    print(f"sudo tc class add dev eth0 parent 1: classid 1:10 htb rate {bandwidth}kbit")
    execute_command(f"sudo tc filter add dev eth0 protocol ip parent 1: prio 1 u32 match ip dst {dest_addr} flowid 1:10")
    print(f"sudo tc filter add dev eth0 protocol ip parent 1: prio 1 u32 match ip dst {dest_addr} flowid 1:10")
    execute_command(f"sudo tc qdisc add dev eth0 parent 1:10 netem loss {loss_rate}% delay {tx_delay}ms limit 20000")
    print(f"sudo tc qdisc add dev eth0 parent 1:10 netem loss {loss_rate}% delay {tx_delay}ms limit 20000")

def extract_time_from_ionadmin(source_node, destination_node):
    """从输出中提取特定时间并返回字符串形式"""
    cmd1 = "ionadmin"
    cmd2 = "l contact"
    output = execute_interactive_commands(cmd1, cmd2)
    if not output:
        return None
    pattern = r"From\s+(\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2})\s+to\s+\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\s+the xmit rate from node (\d+) to node (\d+)"
    matches = re.findall(pattern, output)
    for match in matches:
        from_time = match[0]
        from_node = int(match[1])
        to_node = int(match[2])
        if from_node == source_node and to_node == destination_node:
            return from_time
    print(f"Could not find the target nodes {source_node} to {destination_node} in the output.")
    return None

def execute_interactive_commands(cmd1, cmd2):
    """执行交互式命令"""
    try:
        print("Starting interactive process...")
        process = subprocess.Popen(
            [cmd1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        time.sleep(0.1)
        print(f"Sending command: '{cmd2}'")
        process.stdin.write(cmd2 + "\n")
        process.stdin.flush()
        output, error = process.communicate(timeout=10)
        # print(f"Output:\n{output}")
        if error:
            print(f"Error:\n{error}")
        return output
    except subprocess.TimeoutExpired:
        print("The process did not finish within the timeout period.")
        process.kill()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()


def setup_ltp_span(destination_eid, ltp_sessions, ltp_segment, ltp_block, ltp_aggregation_time, dest_udp_addr):
    """设置 LTP span"""
    cmd = f"c span {destination_eid} {ltp_sessions} {ltp_sessions} {ltp_segment} {ltp_block} {ltp_aggregation_time} {dest_udp_addr}"
    print(cmd)
    execute_interactive_commands("ltpadmin", cmd)

def setup_contact(node_start_time, own_eid, destination_eid, send_rate_B):
    """设置 contact"""
    cmd = f"c contact {node_start_time} {own_eid} {destination_eid} {send_rate_B}"
    print(cmd)
    execute_interactive_commands("ionadmin", cmd)
    cmd = f"c contact {node_start_time} {destination_eid} {own_eid} {send_rate_B}"
    print(cmd)
    execute_interactive_commands("ionadmin", cmd)

def calculate_ltp_sessions(delay, bundle_size, file_size, block_size, trans_rate):
    """计算ltp_sessions数量"""
    total_bundles = math.ceil(file_size / bundle_size)
    bundles_per_block = math.ceil(block_size / bundle_size)
    ltp_blocks = math.ceil(total_bundles / bundles_per_block)
    times = delay/500 + ((block_size + 20) / trans_rate)
    ltp_sessions = math.ceil((times * trans_rate) / (block_size + 20)) + 1
    ltp_sessions = min(ltp_sessions, ltp_blocks + 1, 20)
    return ltp_sessions

def send_bpdriver_command(nbr_of_cycles, source, destination, packet_size, BP_TTL, send_rate_B):
    """构造并发送 bpdriver 命令"""
    i_send_rate_b = f"i{send_rate_B*8}"
    t_TTL = f"t{BP_TTL}"
    # command = f"bpdriver {nbr_of_cycles} {source} {destination} {packet_size} {t_TTL} {i_send_rate_b}"
    command = f"bpdriver {nbr_of_cycles} {source} {destination} {-packet_size} {t_TTL} {i_send_rate_b}"
    print(f"Executing command: {command}")
    send_time = time.time()
    subprocess.run(command, shell=True)
    return send_time

def run_sdrwatch_once_and_record(save_txt_path: str = "sdrwatch_max_total_used.txt", timeout: int = 10) -> int:
    """Run `sdrwatch ion -t` once, parse the numeric value after
    `max total used:` and append a line `TIMESTAMP: VALUE` to `save_txt_path`.

    Returns the parsed integer value, or None if not found/failed.
    """
    import shlex

    command = "sdrwatch ion -t"
    try:
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"运行命令失败: {e}")
        return None

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"max total used:\s*([0-9,]+)", out, re.IGNORECASE)
    if not m:
        print("未找到 'max total used' 字段")
        return None

    try:
        value = int(m.group(1).replace(",", ""))
    except ValueError:
        print("解析值为整数失败")
        return None

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(save_txt_path, 'a') as f:
            f.write(f"{ts}: {value}\n")
    except Exception as e:
        print(f"写入文件失败: {e}")

    return value

def run_bpcounter(source, max_count):
    """运行 bpcounter 命令并捕获输出"""
    command = f"sudo bpcounter {source} {max_count}"
    print(f"Executing command: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    report_lines = []
    report_started = False
    start_marker = "Stopping bpcounter"
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        line = line.strip()
        print(line)
        if start_marker in line:
            report_started = True
        if report_started:
            report_lines.append(line)
    process.wait()
    return "\n".join(report_lines)

def run_bpcounter_and_monitor(source, max_count):
    command = f"sudo bpcounter {source} {max_count}"
    print(f"Executing command: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    report_lines = []

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        line = line.strip()
        print(line)
        report_lines.append(line)

    process.wait()
    stop_time = time.time()
    
    return "\n".join(report_lines), stop_time