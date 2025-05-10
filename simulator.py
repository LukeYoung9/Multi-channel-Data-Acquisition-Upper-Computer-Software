import socket
import threading
import time
import numpy as np
import struct

class DataGenerator:
    def __init__(self):
        self.sample_rate = 100  # 采样率100Hz
        self.t = 0  # 时间计数器
        self.max_t = 1000000  # 时间计数器最大值
        self.last_reset_time = time.time()  # 上次重置时间
        
    def set_sample_rate(self, rate):
        if rate <= 0:
            raise ValueError("采样率必须大于0")
        self.sample_rate = rate
        self.t = 0  # 重置时间计数器
        self.last_reset_time = time.time()
        
    def generate_test_signals(self):
        # 检查是否需要重置时间计数器
        if self.t >= self.max_t:
            self.t = 0  # 重置时间计数器
        # 更新时间计数器，确保连续性
        current_time = time.time()
        elapsed_time = current_time - self.last_reset_time
        expected_samples = int(elapsed_time * self.sample_rate)
        self.t = expected_samples % self.max_t  # 使用取模运算确保连续性
            
        # 生成36通道的测试信号
        signals = []
        try:
            for i in range(36):
                # 不同通道使用不同频率的正弦波
                freq = 1 + i * 0.2  # 频率从1Hz开始，每个通道增加0.2Hz
                amplitude = 1.0 + i * 0.1  # 振幅从1开始，每个通道增加0.1
                signal = amplitude * np.sin(2 * np.pi * freq * self.t / self.sample_rate)
                signals.append(signal)
            self.t += 1
        except Exception as e:
            print(f"生成信号错误: {e}")
            # 发生错误时返回零信号
            signals = [0.0] * 36
        return signals

class DeviceSimulator:
    def __init__(self, host='127.0.0.1', port=8080):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置端口复用选项
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.is_running = False
        self.is_sampling = False
        self.data_generator = DataGenerator()
        
    def start(self):
        print(f"模拟设备启动，等待连接...")
        self.is_running = True
        self.accept_thread = threading.Thread(target=self.accept_connections)
        self.accept_thread.start()
        
    def stop(self):
        self.is_running = False
        self.server_socket.close()
        
    def accept_connections(self):
        while self.is_running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"上位机已连接: {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.start()
            except:
                break
                
    def handle_client(self, client_socket):
        while self.is_running:
            try:
                # 接收二进制命令
                command_data = client_socket.recv(5)  # 最大命令长度为5字节（1字节命令类型 + 4字节参数）
                if not command_data:
                    continue
                    
                # 解析命令类型（第一个字节）
                command_type = command_data[0]
                
                if command_type == 1:  # 设置采样率命令
                    # 确保收到完整的命令（1字节命令类型 + 4字节整数参数）
                    if len(command_data) >= 5:
                        # 解析采样率参数（4字节整数）
                        rate, = struct.unpack('!I', command_data[1:5])
                        print(f"设置采样率为: {rate}Hz")
                        self.data_generator.set_sample_rate(rate)
                    else:
                        print("采样率命令不完整")
                        
                elif command_type == 2:  # 开始采样命令
                    print("收到开始采样指令")
                    self.is_sampling = True
                    sampling_thread = threading.Thread(target=self.send_data, args=(client_socket,))
                    sampling_thread.start()
                    
                elif command_type == 3:  # 停止采样命令
                    print("收到停止采样指令")
                    self.is_sampling = False
                    
            except Exception as e:
                print(f"处理命令错误: {e}")
                break
                
        client_socket.close()
        
    def send_data(self, client_socket):
        # 二进制数据格式：4字节包头(0xAA55AA55) + 4字节包长度 + 36个4字节浮点数
        HEADER = 0xAA55AA55
        CHANNELS = 36
        FLOAT_SIZE = 4
        DATA_LENGTH = CHANNELS * FLOAT_SIZE
        PACKET_SIZE = 8 + DATA_LENGTH  # 包头(4字节) + 包长度(4字节) + 数据
        
        last_send_time = time.time()
        error_count = 0
        max_errors = 3  # 最大连续错误次数
        
        while self.is_sampling and self.is_running:
            try:
                current_time = time.time()
                expected_interval = 1.0 / self.data_generator.sample_rate
                elapsed_time = current_time - last_send_time
                
                if elapsed_time < expected_interval:
                    # 精确控制发送间隔
                    time.sleep(expected_interval - elapsed_time)
                
                # 生成测试数据
                data = self.data_generator.generate_test_signals()
                
                # 构建二进制数据包
                packet = bytearray()
                packet.extend(struct.pack('!I', HEADER))
                packet.extend(struct.pack('!I', DATA_LENGTH))
                
                # 添加36个通道的浮点数据
                for value in data:
                    packet.extend(struct.pack('!f', value))
                
                # 验证数据包大小
                if len(packet) != PACKET_SIZE:
                    raise ValueError(f"数据包大小错误: {len(packet)} != {PACKET_SIZE}")
                
                # 发送数据包
                bytes_sent = client_socket.send(packet)
                if bytes_sent != PACKET_SIZE:
                    raise ValueError(f"数据发送不完整: {bytes_sent} != {PACKET_SIZE}")
                
                last_send_time = time.time()
                error_count = 0  # 重置错误计数
                
            except Exception as e:
                error_count += 1
                print(f"发送数据错误 ({error_count}/{max_errors}): {e}")
                
                if error_count >= max_errors:
                    print("达到最大错误次数，停止发送")
                    break
                    
                # 短暂等待后继续尝试
                time.sleep(0.1)

if __name__ == '__main__':
    # 创建并启动模拟设备
    simulator = DeviceSimulator()
    simulator.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在关闭模拟设备...")
        simulator.stop()