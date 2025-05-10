import sys
import socket
import threading
import time
import csv
import struct
from datetime import datetime
from queue import Queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QScrollArea, QGridLayout, QFileDialog, QSlider, QHBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
import pyqtgraph as pg
import numpy as np
import pandas as pd

class DataReceiver(QObject):
    data_received = pyqtSignal(list)  # 36通道数据接收信号
    
    def __init__(self):
        super().__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置端口复用
        self.is_running = False
        self.is_replaying = False  # 是否正在回放
        self.is_connected = False  # 是否已连接
        self.data_buffer = [[] for _ in range(36)]  # 36个通道的数据缓冲区
        self.data_queue = Queue(maxsize=1000)  # 数据队列，限制大小为1000
        self.max_points = 1000  # 限制每个通道显示的数据点数
        self.sample_rate = 100  # 默认采样率100Hz
        self.replay_data = None  # 回放数据
        self.replay_index = 0
        self.replay_speed = 1.0  # 回放速度  # 回放索引

    def connect_to_device(self, ip, port):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置端口复用
            self.socket.connect((ip, port))
            self.is_connected = True
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            self.is_connected = False
            return False
            
    def disconnect_device(self):
        try:
            if self.is_running:
                self.stop_sampling()
            if self.is_connected:
                self.socket.close()
                self.is_connected = False
            # 清空数据缓冲区和队列
            self.data_buffer = [[] for _ in range(36)]
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    pass
            return True
        except Exception as e:
            print(f"断开连接失败: {e}")
            return False
            
    def start_sampling(self):
        # 发送采样率和开始命令（二进制格式）
        # 命令格式：1字节命令类型 + 4字节整数参数
        # 命令类型：1=设置采样率，2=开始采样
        self.socket.send(struct.pack('!BI', 1, self.sample_rate))  # 设置采样率命令
        self.socket.send(struct.pack('!B', 2))  # 开始采样命令
        self.is_running = True
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.start()

    def set_sample_rate(self, rate):
        if rate <= 0:
            raise ValueError("采样率必须大于0")
        # 停止当前采样
        if self.is_running:
            self.stop_sampling()
            time.sleep(0.1)  # 等待停止完成
        
        # 清空数据缓冲区和队列
        self.data_buffer = [[] for _ in range(36)]
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                pass
        
        self.sample_rate = rate

    def stop_sampling(self):
        self.is_running = False
        # 命令类型：3=停止采样
        self.socket.send(struct.pack('!B', 3))  # 停止采样命令
        
    def receive_data(self):
        # 二进制数据格式：4字节包头(0xAA55AA55) + 4字节包长度 + 36个4字节浮点数
        # 每个数据包总长度：4 + 4 + 36*4 = 152字节
        HEADER = 0xAA55AA55
        FLOAT_SIZE = 4
        CHANNELS = 36
        HEADER_SIZE = 8  # 包头(4字节) + 包长度(4字节)
        PACKET_SIZE = HEADER_SIZE + CHANNELS * FLOAT_SIZE
        BUFFER_SIZE = PACKET_SIZE * 10  # 增大接收缓冲区大小
        
        buffer = bytearray()
        last_process_time = time.time()  # 记录上次处理时间
        batch_values = []  # 批量处理数据
        
        while self.is_running:
            try:
                # 增大接收缓冲区
                chunk = self.socket.recv(BUFFER_SIZE)
                if not chunk:
                    continue
                    
                buffer.extend(chunk)
                
                # 处理完整的数据包
                while len(buffer) >= PACKET_SIZE:
                    # 检查包头
                    header, length = struct.unpack('!II', buffer[:HEADER_SIZE])
                    if header != HEADER:
                        # 包头不匹配，尝试重新同步
                        next_pos = buffer[1:].find(struct.pack('!I', HEADER))
                        if next_pos == -1:
                            buffer = bytearray()  # 清空缓冲区
                        else:
                            buffer = buffer[next_pos+1:]  # 从可能的包头位置开始
                        continue
                    
                    # 检查包长度
                    if length != CHANNELS * FLOAT_SIZE:
                        buffer = buffer[HEADER_SIZE:]  # 跳过当前包头
                        continue
                    
                    # 确保有足够的数据
                    if len(buffer) < PACKET_SIZE:
                        break
                    
                    # 解析36个浮点数
                    values = []
                    data = buffer[HEADER_SIZE:HEADER_SIZE + CHANNELS * FLOAT_SIZE]
                    values = list(struct.unpack('!' + 'f' * CHANNELS, data))
                    
                    # 批量收集数据
                    batch_values.append(values)
                    
                    # 移除已处理的数据包
                    buffer = buffer[PACKET_SIZE:]
                    
                    # 当积累了足够的数据或时间间隔达到要求时进行批量处理
                    current_time = time.time()
                    time_diff = current_time - last_process_time
                    expected_interval = 1.0 / self.sample_rate
                    
                    if time_diff >= expected_interval and batch_values:
                        # 批量处理数据
                        for values in batch_values:
                            if not self.data_queue.full():
                                self.data_queue.put(values)
                            
                            # 更新数据缓冲区
                            for i, value in enumerate(values):
                                self.data_buffer[i].append(value)
                                if len(self.data_buffer[i]) > self.max_points:
                                    self.data_buffer[i].pop(0)
                        
                        batch_values = []  # 清空批处理缓存
                        last_process_time = current_time
                        
            except Exception as e:
                print(f"数据接收错误: {e}")
                if not self.is_running:
                    break
                
    def save_data(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Time'] + [f'Channel_{i+1}' for i in range(36)]
            writer.writerow(headers)
            # 获取所有通道中最大的数据点数
            max_length = max(len(channel) for channel in self.data_buffer)
            for i in range(max_length):
                row = [i]
                for channel in self.data_buffer:
                    row.append(channel[i] if i < len(channel) else '')
                writer.writerow(row)
                
    def load_data(self, filename):
        try:
            df = pd.read_csv(filename)
            self.replay_data = df.iloc[:, 1:].values  # 跳过Time列
            self.replay_index = 0
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
            
    def start_replay(self, speed=1.0):
        if self.replay_data is None:
            return False
        # 重置回放状态
        self.stop_replay()
        time.sleep(0.1)  # 确保之前的回放线程已经停止
        
        # 清空数据队列和缓冲区
        self.data_buffer = [[] for _ in range(36)]
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                pass
                
        self.replay_index = 0
        self.replay_speed = speed
        self.is_replaying = True
        self.replay_thread = threading.Thread(target=self.replay_data_thread)
        self.replay_thread.daemon = True  # 设置为守护线程
        self.replay_thread.start()
        return True
        
    def stop_replay(self):
        if self.is_replaying:
            self.is_replaying = False
            time.sleep(0.1)  # 等待线程停止
            # 清空数据队列和缓冲区
            self.data_buffer = [[] for _ in range(36)]
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    pass
        
    def replay_data_thread(self):
        last_update_time = time.time()
        target_interval = 1.0 / (self.sample_rate * self.replay_speed)
        accumulated_error = 0.0  # 累积时间误差
        batch_values = []  # 批量处理数据
        batch_size = 10  # 每批处理的数据包数量
        
        while self.is_replaying and self.replay_index < len(self.replay_data):
            current_time = time.time()
            time_diff = current_time - last_update_time
            
            # 考虑累积误差调整时间间隔
            adjusted_interval = target_interval - accumulated_error
            
            if time_diff >= adjusted_interval:
                try:
                    # 收集一批数据
                    while len(batch_values) < batch_size and self.replay_index < len(self.replay_data):
                        values = self.replay_data[self.replay_index].tolist()
                        batch_values.append(values)
                        self.replay_index += 1
                    
                    # 批量处理数据
                    if batch_values:
                        for values in batch_values:
                            # 使用非阻塞方式管理队列
                            if self.data_queue.full():
                                try:
                                    self.data_queue.get_nowait()
                                except:
                                    pass
                            self.data_queue.put_nowait(values)
                            
                            # 更新数据缓冲区
                            for i, value in enumerate(values):
                                self.data_buffer[i].append(value)
                                if len(self.data_buffer[i]) > self.max_points:
                                    self.data_buffer[i].pop(0)
                        
                        # 发送数据更新信号
                        self.data_received.emit(batch_values[-1])
                        batch_values = []  # 清空批处理缓存
                    
                    # 计算并累积时间误差
                    actual_interval = time.time() - last_update_time
                    accumulated_error += actual_interval - target_interval
                    accumulated_error *= 0.9  # 逐渐衰减误差，防止累积过大
                    
                    last_update_time = time.time()
                except Exception as e:
                    print(f"回放数据处理错误: {e}")
                    time.sleep(0.001)  # 发生错误时短暂暂停
            else:
                # 精确控制时间间隔，使用较小的睡眠时间
                sleep_time = min(0.001, adjusted_interval - time_diff)
                if sleep_time > 0:
                    time.sleep(sleep_time)  # 使用当前回放速度

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('数据采集上位机')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.process_queue)
        self.update_timer.setInterval(100)  # 10Hz的更新频率，降低UI刷新频率以提高性能
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 创建控件
        self.ip_input = QLineEdit('127.0.0.1')
        self.port_input = QLineEdit('8080')
        self.sample_rate_input = QLineEdit('100')
        # 创建控件
        self.connect_btn = QPushButton('连接设备')
        self.disconnect_btn = QPushButton('断开连接')
        self.start_btn = QPushButton('开始采样')
        self.stop_btn = QPushButton('停止采样')
        self.save_btn = QPushButton('保存数据')
        self.load_btn = QPushButton('加载数据')
        self.replay_btn = QPushButton('开始回放')
        self.stop_replay_btn = QPushButton('停止回放')
        
        # 创建回放速度滑动条
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_label = QLabel('回放速度: 1.0x')
        
        # 添加控件到布局
        layout.addWidget(QLabel('IP地址:'))
        layout.addWidget(self.ip_input)
        layout.addWidget(QLabel('端口:'))
        layout.addWidget(self.port_input)
        layout.addWidget(QLabel('采样率(Hz):'))
        layout.addWidget(self.sample_rate_input)
        
        # 创建水平布局用于按钮
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.disconnect_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.replay_btn)
        button_layout.addWidget(self.stop_replay_btn)
        layout.addLayout(button_layout)
        
        # 添加回放速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        layout.addLayout(speed_layout)
        
        # 创建36个通道的图表
        self.plot_widgets = []
        self.plot_curves = []
        self.data_x = [[] for _ in range(36)]
        self.data_y = [[] for _ in range(36)]
        
        # 创建网格布局来显示图表
        grid_layout = QGridLayout()
        for i in range(36):
            plot_widget = pg.PlotWidget(title=f'Channel {i+1}')
            plot_widget.setBackground('w')
            plot_curve = plot_widget.plot(pen=(i*7 % 255, 255-(i*7 % 255), i*13 % 255))
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)
            grid_layout.addWidget(plot_widget, i//6, i%6)
        
        # 创建滚动区域来容纳所有图表
        scroll_widget = QWidget()
        scroll_widget.setLayout(grid_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # 初始化数据接收器
        self.receiver = DataReceiver()
        self.receiver.data_received.connect(self.update_plot)
        
        # 连接信号和槽
        self.connect_btn.clicked.connect(self.connect_device)
        self.disconnect_btn.clicked.connect(self.disconnect_device)
        self.start_btn.clicked.connect(self.start_sampling)
        self.stop_btn.clicked.connect(self.stop_sampling)
        self.save_btn.clicked.connect(self.save_data)
        self.load_btn.clicked.connect(self.load_data)
        self.replay_btn.clicked.connect(self.start_replay)
        self.stop_replay_btn.clicked.connect(self.stop_replay)
        self.speed_slider.valueChanged.connect(self.update_speed)
        
        # 初始化按钮状态
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.stop_replay_btn.setEnabled(False)
        
    def connect_device(self):
        ip = self.ip_input.text()
        port = int(self.port_input.text())
        if self.receiver.connect_to_device(ip, port):
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.ip_input.setEnabled(False)
            self.port_input.setEnabled(False)
            
    def disconnect_device(self):
        if self.receiver.disconnect_device():
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.ip_input.setEnabled(True)
            self.port_input.setEnabled(True)
            # 清空所有图表
            for i in range(36):
                self.data_x[i].clear()
                self.data_y[i].clear()
                self.plot_curves[i].setData(self.data_x[i], self.data_y[i])
            
    def start_sampling(self):
        try:
            rate = int(self.sample_rate_input.text())
            if rate <= 0:
                raise ValueError("采样率必须大于0")
            self.receiver.set_sample_rate(rate)
            self.receiver.start_sampling()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.update_timer.start()
        except ValueError as e:
            print(f"采样率设置错误: {e}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.update_timer.start()
        
    def stop_sampling(self):
        self.receiver.stop_sampling()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.update_timer.stop()
        
    def save_data(self):
        filename, _ = QFileDialog.getSaveFileName(self, '保存数据', f'data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 'CSV文件 (*.csv)')
        if filename:
            self.receiver.save_data(filename)
            
    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, '加载数据', '', 'CSV文件 (*.csv)')
        if filename and self.receiver.load_data(filename):
            self.replay_btn.setEnabled(True)
            self.stop_replay_btn.setEnabled(False)
            
    def start_replay(self):
        speed = self.speed_slider.value() / 10.0
        if self.receiver.start_replay(speed):
            self.replay_btn.setEnabled(False)
            self.stop_replay_btn.setEnabled(True)
            self.start_btn.setEnabled(False)
            self.connect_btn.setEnabled(False)
            
    def stop_replay(self):
        self.receiver.stop_replay()
        self.replay_btn.setEnabled(True)
        self.stop_replay_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.connect_btn.setEnabled(True)
        
    def update_speed(self, value):
        speed = value / 10.0
        self.receiver.replay_speed = speed  # 更新DataReceiver中的回放速度
        print(f"回放速度设置为: {speed}")
        self.speed_label.setText(f'回放速度: {speed:.1f}x')
        
    def process_queue(self):
        # 批量处理队列中的数据
        batch_size = 10  # 每次处理的数据包数量
        batch_data = []
        
        # 收集一批数据
        for _ in range(batch_size):
            if self.receiver.data_queue.empty():
                break
            try:
                values = self.receiver.data_queue.get_nowait()
                batch_data.append(values)
            except:
                break
        
        # 批量更新图表
        if batch_data:
            for values in batch_data:
                self.update_plot(values, update_display=False)
            # 最后一次性更新显示
            for i in range(36):
                self.plot_curves[i].setData(self.data_x[i], self.data_y[i])
    
    def update_plot(self, values, update_display=True):
        for i, value in enumerate(values):
            self.data_x[i].append(len(self.data_x[i]))
            self.data_y[i].append(value)
            # 限制显示的数据点数
            if len(self.data_x[i]) > self.receiver.max_points:
                self.data_x[i].pop(0)
                self.data_y[i].pop(0)
            # 仅在需要时更新显示
            if update_display:
                self.plot_curves[i].setData(self.data_x[i], self.data_y[i])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())