#!/usr/bin/env python3
"""
real_time_audio_spatializer_simple.py
实时音频空间化系统 - 简单重叠相加版
取消汉宁窗重叠，使用简单的重叠相加法
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from scipy import signal
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# ==================== HRTF 类 ====================
class HRTF:
    """HRTF数据类"""
    def __init__(self, left_ir, right_ir, azimuth, elevation, sample_rate):
        self.left_ir = left_ir
        self.right_ir = right_ir
        self.azimuth = azimuth
        self.elevation = elevation
        self.sample_rate = sample_rate

class HRTFDatabase:
    """HRTF数据库管理"""
    def __init__(self, sample_rate=48000):
        self.hrtfs = []
        self.sample_rate = sample_rate
        self.positions = []
        
    def load_from_folder(self, folder_path):
        """从文件夹加载HRTF文件"""
        if not os.path.exists(folder_path):
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        if not wav_files:
            raise ValueError(f"在 {folder_path} 中未找到WAV文件")
        
        print(f"找到 {len(wav_files)} 个WAV文件")
        
        for wav_file in wav_files:
            try:
                # 解析方位角和仰角
                filename = os.path.basename(wav_file)
                az, el = self._parse_filename(filename)
                
                print(f"加载文件: {filename}, 位置: ({az}, {el})")
                
                # 加载音频文件
                data, fs = sf.read(wav_file)
                
                # 重采样到目标采样率
                if fs != self.sample_rate:
                    num_samples = int(len(data) * self.sample_rate / fs)
                    data = signal.resample(data, num_samples)
                
                # 确保是立体声
                if data.ndim == 1:
                    data = np.column_stack([data, data])
                elif data.shape[1] > 2:
                    data = data[:, :2]
                
                # 创建HRTF对象
                hrtf = HRTF(
                    left_ir=data[:, 0],
                    right_ir=data[:, 1],
                    azimuth=az,
                    elevation=el,
                    sample_rate=self.sample_rate
                )
                
                self.hrtfs.append(hrtf)
                self.positions.append([az, el])
                
            except Exception as e:
                print(f"警告: 加载文件 {wav_file} 失败: {e}")
        
        if not self.hrtfs:
            raise ValueError("未能加载任何HRTF文件")
        
        self.positions = np.array(self.positions)
        print(f"成功加载 {len(self.hrtfs)} 个HRTF")
        return True
        
    def _parse_filename(self, filename):
        """解析文件名获取方位角和仰角"""
        # 去掉扩展名
        name = os.path.splitext(filename)[0]
        
        # 尝试多种模式
        patterns = [
            r'azi_([-\d,.]+)_ele_([-\d,.]+)',
            r'([-\d,.]+)_([-\d,.]+)',
            r'a([-\d,.]+)e([-\d,.]+)',
            r'azi([-\d,.]+)ele([-\d,.]+)',
            r'([-\d,.]+)deg_([-\d,.]+)deg',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                try:
                    azi_str = match.group(1).replace(',', '.')
                    ele_str = match.group(2).replace(',', '.')
                    az = float(azi_str)
                    el = float(ele_str)
                    return az, el
                except ValueError:
                    continue
        
        print(f"警告: 无法解析文件名 {filename}, 使用默认位置 (0, 0)")
        return 0.0, 0.0

# ==================== 简单重叠相加卷积处理器 ====================
class SimpleOverlapAddConvolutionProcessor:
    """重叠相加卷积处理器"""
    
    def __init__(self, hrtf, block_size=1024):
        """
        初始化处理器
        hrtf: HRTF对象
        block_size: 块大小
        """
        self.hrtf = hrtf
        self.block_size = block_size
        self.hrtf_len = len(hrtf.left_ir)
        
        # 重叠大小（HRTF长度-1）
        self.overlap = self.hrtf_len - 1
        
        # 计算FFT大小（确保足够大）
        self.fft_size = 2 ** int(np.ceil(np.log2(self.block_size + self.hrtf_len - 1)))
        
        # 计算HRTF的频域表示
        self.hrtf_left_fft = np.fft.rfft(hrtf.left_ir, n=self.fft_size)
        self.hrtf_right_fft = np.fft.rfft(hrtf.right_ir, n=self.fft_size)
        
        # 重叠缓冲区
        self.overlap_buffer_left = np.zeros(self.overlap)
        self.overlap_buffer_right = np.zeros(self.overlap)
        
        # 输入缓冲区（用于处理不满一个块的情况）
        self.input_buffer = np.zeros(self.block_size)
        self.input_buffer_pos = 0
        
        # 当前权重（用于平滑过渡）
        self.current_weight = 0.0
        self.target_weight = 0.0
    
    def set_weight(self, target_weight):
        """设置目标权重"""
        self.target_weight = target_weight
    
    def update_weight(self):
        """更新当前权重，平滑过渡到目标权重"""
        if abs(self.current_weight - self.target_weight) < 0.001:
            self.current_weight = self.target_weight
        else:
            # 使用指数平滑过渡
            self.current_weight = self.current_weight * 0.9 + self.target_weight * 0.1
        return self.current_weight
    
    def process_block(self, audio_block):
        """
        处理一个音频块，返回立体声输出
        音频块大小应为block_size
        """
        # 确保输入块大小正确
        if len(audio_block) != self.block_size:
            # 如果输入块大小不正确，使用输入缓冲区
            if len(audio_block) < self.block_size:
                # 将数据添加到输入缓冲区
                remaining = min(len(audio_block), self.block_size - self.input_buffer_pos)
                self.input_buffer[self.input_buffer_pos:self.input_buffer_pos+remaining] = audio_block[:remaining]
                self.input_buffer_pos += remaining
                
                if self.input_buffer_pos < self.block_size:
                    # 缓冲区还没满，返回空
                    return np.zeros((0, 2))
                
                # 缓冲区已满，处理完整块
                audio_block = self.input_buffer.copy()
                self.input_buffer_pos = 0
            else:
                # 截断
                audio_block = audio_block[:self.block_size]
        
        # 更新权重
        self.update_weight()
        
        # 如果权重接近0，跳过处理以节省计算
        if self.current_weight < 0.001:
            return np.zeros((self.block_size, 2))
        
        # 对输入块进行FFT
        input_fft = np.fft.rfft(audio_block, n=self.fft_size)
        
        # 频域相乘
        output_left_fft = input_fft * self.hrtf_left_fft
        output_right_fft = input_fft * self.hrtf_right_fft
        
        # 逆FFT得到时域输出
        output_left_full = np.fft.irfft(output_left_fft, n=self.fft_size)
        output_right_full = np.fft.irfft(output_right_fft, n=self.fft_size)
        
        # 重叠相加：当前块的前overlap部分 + 前一个块的重叠缓冲区
        output_left = np.zeros(self.block_size)
        output_right = np.zeros(self.block_size)
        
        if self.overlap > 0:
            output_left[:self.overlap] = self.overlap_buffer_left + output_left_full[:self.overlap]
            output_right[:self.overlap] = self.overlap_buffer_right + output_right_full[:self.overlap]
        
        # 剩余部分
        if self.block_size > self.overlap:
            output_left[self.overlap:] = output_left_full[self.overlap:self.block_size]
            output_right[self.overlap:] = output_right_full[self.overlap:self.block_size]
        
        # 保存重叠部分用于下一个块
        if self.overlap > 0:
            self.overlap_buffer_left = output_left_full[self.block_size:self.block_size+self.overlap].copy()
            self.overlap_buffer_right = output_right_full[self.block_size:self.block_size+self.overlap].copy()
        
        # 应用权重
        output_left = output_left * self.current_weight
        output_right = output_right * self.current_weight
        
        # 返回当前输出块
        return np.column_stack([output_left, output_right])
    
    def flush(self):
        """处理剩余的尾音"""
        # 处理输入缓冲区中剩余的数据
        if self.input_buffer_pos > 0:
            # 对剩余数据进行零填充
            padded_block = np.zeros(self.block_size)
            padded_block[:self.input_buffer_pos] = self.input_buffer[:self.input_buffer_pos]
            output = self.process_block(padded_block)
            self.input_buffer_pos = 0
            return output
        
        return np.zeros((0, 2))
    
    def reset_state(self):
        """重置处理器状态"""
        self.overlap_buffer_left.fill(0)
        self.overlap_buffer_right.fill(0)
        self.input_buffer.fill(0)
        self.input_buffer_pos = 0
        self.current_weight = 0.0
        self.target_weight = 0.0

# ==================== 实时音频处理器 ====================
class RealTimeAudioProcessor:
    """实时音频处理器 - 基于分块频域卷积方法"""
    def __init__(self, sample_rate=48000, block_size=1024, max_hrtfs=9):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.max_hrtfs = max_hrtfs
        
        self.hrtf_db = None
        self.audio_data = None
        self.current_position = 0
        self.volume = 1.0
        self.target_azimuth = 0.0
        self.target_elevation = 0.0
        self.weights = None
        self.prev_weights = None
        
        # 为所有HRTF创建处理器
        self.all_processors = []
        
        # 音频流
        self.stream = None
        self.is_playing = False
        
        # 延迟补偿
        self.delay_compensation = 0
        
        # 性能统计
        self.processing_times = []
        self.avg_processing_time = 0
        
        # 输入缓冲区（用于处理不满一个块的情况）
        self.input_buffer = np.zeros(block_size)
        self.buffer_pos = 0
        
        # 平滑过渡相关
        self.smooth_transition_enabled = True
        
    def set_hrtf_database(self, hrtf_db):
        """设置HRTF数据库"""
        self.hrtf_db = hrtf_db
        
        # 为每个HRTF创建处理器
        self.all_processors = []
        
        if hrtf_db and hrtf_db.hrtfs:
            for hrtf in hrtf_db.hrtfs:
                processor = SimpleOverlapAddConvolutionProcessor(hrtf, self.block_size)
                self.all_processors.append(processor)
            
            print(f"为 {len(self.all_processors)} 个HRTF创建了处理器")
            
            # 计算最大HRTF长度用于延迟补偿
            max_hrtf_len = max(len(hrtf.left_ir) for hrtf in hrtf_db.hrtfs)
            self.delay_compensation = max_hrtf_len - 1
            
    def set_audio_data(self, audio_data):
        """设置音频数据"""
        self.audio_data = audio_data
    
    def calculate_weights(self, azimuth, elevation):
        """计算HRTF权重，限制数量在max_hrtfs以内"""
        if self.hrtf_db is None:
            return np.array([])
        
        positions = self.hrtf_db.positions
        num_hrtfs = len(positions)
        weights = np.zeros(num_hrtfs)
        
        # 计算球面距离
        distances = []
        for i in range(num_hrtfs):
            # 方位角差（处理0-360度循环）
            az_diff = abs(positions[i, 0] - azimuth)
            if az_diff > 180:
                az_diff = 360 - az_diff
            
            el_diff = abs(positions[i, 1] - elevation)
            
            # 综合距离
            distance = np.sqrt((az_diff/15)**2 + (el_diff)**2)
            distances.append(distance)
            
            # 高斯权重
            if distance < 45:  # 阈值角度45度
                weights[i] = np.exp(-distance**2 / 200)
        
        # 归一化
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        
        # 选择权重最大的前max_hrtfs个
        if np.sum(weights > 0.001) > self.max_hrtfs:
            # 获取权重最大的索引
            top_indices = np.argsort(weights)[-self.max_hrtfs:]
            
            # 创建新的权重数组，只保留选中的HRTF
            new_weights = np.zeros(num_hrtfs)
            new_weights[top_indices] = weights[top_indices]
            
            # 重新归一化
            total_weight = np.sum(new_weights)
            if total_weight > 0:
                new_weights = new_weights / total_weight
            else:
                # 如果都没有权重，使用最近的一个
                closest_idx = np.argmin(distances)
                new_weights = np.zeros(num_hrtfs)
                new_weights[closest_idx] = 1.0
            
            weights = new_weights
        
        # 只保留显著权重（>0.1%）
        weights[weights < 0.001] = 0
        
        # 重新归一化
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # 如果没有有效的HRTF，使用最近的一个
            closest_idx = np.argmin(distances)
            weights = np.zeros(num_hrtfs)
            weights[closest_idx] = 1.0
        
        return weights
    
    def update_position(self, azimuth, elevation):
        """更新声源位置 - 平滑过渡版"""
        start_time = time.time()
        
        self.target_azimuth = azimuth
        self.target_elevation = elevation
        
        # 保存之前的权重
        if self.weights is not None:
            self.prev_weights = self.weights.copy()
        
        # 计算新的权重
        self.weights = self.calculate_weights(azimuth, elevation)
        
        # 更新所有处理器的目标权重
        if self.all_processors and len(self.weights) == len(self.all_processors):
            for i, processor in enumerate(self.all_processors):
                processor.set_weight(self.weights[i])
        
        # 更新性能统计
        processing_time = (time.time() - start_time) * 1000  # 毫秒
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)
        self.avg_processing_time = np.mean(self.processing_times)
        
        active_count = np.sum(self.weights > 0.001)
        if active_count > 0:
            print(f"位置更新: ({azimuth}°, {elevation}°), 活跃HRTF: {active_count}, "
                  f"处理时间: {processing_time:.2f}ms")
        
        return active_count
    
    def audio_callback(self, outdata, frames, time_info, status):
        """音频回调函数 - 简单重叠相加版"""
        if status:
            print(f"音频状态: {status}")
        
        if not self.is_playing or self.audio_data is None:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return
        
        # 检查是否还有数据
        if self.current_position >= len(self.audio_data):
            self.is_playing = False
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return
        
        start_time = time.time()
        
        # 获取当前帧（应用延迟补偿）
        read_pos = max(0, self.current_position - self.delay_compensation)
        end_pos = min(read_pos + frames, len(self.audio_data))
        actual_frames = end_pos - read_pos
        
        if actual_frames <= 0:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return
        
        # 读取音频数据
        audio_block = self.audio_data[read_pos:end_pos]
        
        # 使用输入缓冲区处理不满一个块的情况
        if actual_frames < frames:
            # 将数据添加到缓冲区
            remaining = min(len(audio_block), frames - self.buffer_pos)
            self.input_buffer[self.buffer_pos:self.buffer_pos+remaining] = audio_block[:remaining]
            self.buffer_pos += remaining
            
            if self.buffer_pos < frames:
                # 缓冲区还没满，返回空
                outdata[:] = np.zeros((frames, 2), dtype=np.float32)
                return
            
            # 缓冲区已满，处理完整块
            audio_block = self.input_buffer.copy()
            self.buffer_pos = 0
        else:
            # 如果数据足够，直接使用
            if len(audio_block) > frames:
                audio_block = audio_block[:frames]
            elif len(audio_block) < frames:
                # 如果数据不够，补零
                padded_block = np.zeros(frames)
                padded_block[:len(audio_block)] = audio_block
                audio_block = padded_block
        
        # 初始化输出
        output = np.zeros((frames, 2))
        
        # 如果有HRTF处理器，进行处理
        if self.all_processors and len(self.all_processors) > 0:
            # 为每个HRTF处理器处理当前块
            for i, processor in enumerate(self.all_processors):
                # 每个处理器独立处理相同的音频块
                conv_output = processor.process_block(audio_block)
                
                # 确保输出大小正确
                if len(conv_output) < frames:
                    # 补零
                    padded_output = np.zeros((frames, 2))
                    padded_output[:len(conv_output)] = conv_output
                    conv_output = padded_output
                elif len(conv_output) > frames:
                    # 截断
                    conv_output = conv_output[:frames]
                
                # 直接相加（权重已在处理器内部应用）
                output += conv_output
        
        # 如果没有有效的输出，直接播放原始音频
        if np.max(np.abs(output)) < 1e-6:
            output[:, 0] = audio_block
            output[:, 1] = audio_block
        
        # 应用音量
        output = output * self.volume
        
        # 防止削波
        max_val = np.max(np.abs(output))
        if max_val > 0.9:
            output = output / max_val * 0.9
        
        # 输出到音频设备
        outdata[:] = output.astype(np.float32)
        
        # 更新播放位置
        self.current_position += frames
        
        # 更新性能统计
        processing_time = (time.time() - start_time) * 1000  # 毫秒
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def start_playback(self):
        """开始播放"""
        if self.audio_data is None:
            print("错误: 未加载音频数据")
            return False
        
        if self.stream is not None:
            self.stream.close()
        
        self.is_playing = True
        self.current_position = 0
        
        # 重置输入缓冲区
        self.input_buffer.fill(0)
        self.buffer_pos = 0
        
        # 重置所有处理器状态
        for processor in self.all_processors:
            processor.reset_state()
        
        try:
            # 创建音频流
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.block_size,
                callback=self.audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            print(f"开始播放，块大小: {self.block_size}，重叠相加法")
            return True
            
        except Exception as e:
            print(f"启动音频流失败: {e}")
            self.is_playing = False
            return False
    
    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        # 刷新所有处理器
        for processor in self.all_processors:
            processor.flush()
        
        print("播放停止")
    
    def get_progress(self):
        """获取播放进度"""
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0
        return min(1.0, self.current_position / len(self.audio_data))
    
    def get_audio_duration(self):
        """获取音频时长（秒）"""
        if self.audio_data is None:
            return 0
        return len(self.audio_data) / self.sample_rate
    
    def get_avg_processing_time(self):
        """获取平均处理时间"""
        if len(self.processing_times) == 0:
            return 0
        return np.mean(self.processing_times[-50:])

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.block_size = 1024
        
        # 初始化处理器
        self.processor = RealTimeAudioProcessor(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            max_hrtfs=9
        )
        
        # 初始化UI
        self.init_ui()
        
        # 启动进度更新定时器
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)
        
        # 性能监控定时器
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance)
        self.perf_timer.start(1000)
        
        # 音频数据
        self.audio_data = None
        self.hrtf_db = None
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("实时音频处理器 - 基于分块频域卷积方法")
        self.setGeometry(100, 100, 600, 450)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 信息显示面板
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        layout = QGridLayout()
        
        # 音频文件选择
        self.audio_path_label = QLabel("未选择音频文件")
        self.audio_path_label.setMinimumWidth(200)
        audio_browse_btn = QPushButton("浏览音频...")
        audio_browse_btn.clicked.connect(self.browse_audio_file)
        
        # HRTF文件夹选择
        self.hrtf_path_label = QLabel("未选择HRTF文件夹")
        self.hrtf_path_label.setMinimumWidth(200)
        hrtf_browse_btn = QPushButton("浏览HRTF...")
        hrtf_browse_btn.clicked.connect(self.browse_hrtf_folder)
        
        # 位置控制
        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(-180, 180)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.valueChanged.connect(self.update_position)
        self.azimuth_label = QLabel("方位角: 0°")
        
        # 方位角刻度标签
        azimuth_label_layout = QHBoxLayout()
        azimuth_label_layout.addWidget(QLabel("-180°"))
        azimuth_label_layout.addStretch()
        azimuth_label_layout.addWidget(QLabel("0°"))
        azimuth_label_layout.addStretch()
        azimuth_label_layout.addWidget(QLabel("180°"))
        
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.setValue(0)
        self.elevation_slider.valueChanged.connect(self.update_position)
        self.elevation_label = QLabel("仰角: 0°")
        
        # 音量控制
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.volume_label = QLabel("音量: 100%")
        
        # 块大小选择
        self.block_combo = QComboBox()
        self.block_combo.addItems(["256", "512", "1024", "2048"])
        self.block_combo.setCurrentText("1024")
        self.block_combo.currentTextChanged.connect(self.update_block_size)
        
        # HRTF数量限制
        self.hrtf_limit_combo = QComboBox()
        self.hrtf_limit_combo.addItems(["3", "5", "7", "9", "12"])
        self.hrtf_limit_combo.setCurrentText("9")
        self.hrtf_limit_combo.currentTextChanged.connect(self.update_hrtf_limit)
        
        # 平滑过渡开关
        self.smooth_transition_checkbox = QCheckBox("平滑过渡")
        self.smooth_transition_checkbox.setChecked(True)
        self.smooth_transition_checkbox.stateChanged.connect(self.update_smooth_transition)
        
        # 控制按钮
        self.play_btn = QPushButton("开始播放")
        self.play_btn.clicked.connect(self.start_playback)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("停止播放")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        
        self.test_btn = QPushButton("测试声音")
        self.test_btn.clicked.connect(self.test_audio)
        
        # 布局
        row = 0
        layout.addWidget(QLabel("音频文件:"), row, 0)
        layout.addWidget(self.audio_path_label, row, 1, 1, 2)
        layout.addWidget(audio_browse_btn, row, 3)
        
        row += 1
        layout.addWidget(QLabel("HRTF文件夹:"), row, 0)
        layout.addWidget(self.hrtf_path_label, row, 1, 1, 2)
        layout.addWidget(hrtf_browse_btn, row, 3)
        
        row += 1
        layout.addWidget(QLabel("方位角:"), row, 0)
        layout.addWidget(self.azimuth_slider, row, 1, 1, 2)
        layout.addWidget(self.azimuth_label, row, 3)
        
        row += 1
        layout.addLayout(azimuth_label_layout, row, 1, 1, 2)
        
        row += 1
        layout.addWidget(QLabel("仰角:"), row, 0)
        layout.addWidget(self.elevation_slider, row, 1, 1, 2)
        layout.addWidget(self.elevation_label, row, 3)
        
        row += 1
        layout.addWidget(QLabel("音量:"), row, 0)
        layout.addWidget(self.volume_slider, row, 1, 1, 2)
        layout.addWidget(self.volume_label, row, 3)
        
        row += 1
        layout.addWidget(QLabel("块大小:"), row, 0)
        layout.addWidget(self.block_combo, row, 1)
        layout.addWidget(QLabel("HRTF限制:"), row, 2)
        layout.addWidget(self.hrtf_limit_combo, row, 3)
        
        row += 1
        layout.addWidget(self.smooth_transition_checkbox, row, 0)
        
        row += 1
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.test_btn)
        layout.addLayout(button_layout, row, 0, 1, 4)
        
        panel.setLayout(layout)
        return panel
    
    def create_info_panel(self):
        """创建信息显示面板"""
        panel = QGroupBox("状态信息")
        layout = QVBoxLayout()
        
        # 音频信息
        self.audio_info_label = QLabel("音频: 未加载")
        layout.addWidget(self.audio_info_label)
        
        # HRTF信息
        self.hrtf_info_label = QLabel("HRTF: 未加载")
        layout.addWidget(self.hrtf_info_label)
        
        # 当前位置信息
        self.position_info_label = QLabel("当前位置: 方位角 0°, 仰角 0°")
        layout.addWidget(self.position_info_label)
        
        # 活跃HRTF信息
        self.active_hrtf_label = QLabel("活跃HRTF: 0/9个")
        layout.addWidget(self.active_hrtf_label)
        
        # 性能信息
        self.perf_label = QLabel("处理延迟: 0.0ms | 卷积延迟: 0采样点")
        layout.addWidget(self.perf_label)
        
        # 播放进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(QLabel("播放进度:"))
        layout.addWidget(self.progress_bar)
        
        # 卷积方法信息
        self.method_label = QLabel("卷积方法: 简单重叠相加")
        layout.addWidget(self.method_label)
        
        panel.setLayout(layout)
        return panel
    
    def browse_audio_file(self):
        """浏览音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.flac)"
        )
        
        if file_path:
            try:
                filename = os.path.basename(file_path)
                self.audio_path_label.setText(filename)
                
                # 加载音频
                data, fs = sf.read(file_path)
                
                # 转换为单声道并重采样
                if data.ndim > 1:
                    data = data.mean(axis=1)
                
                if fs != self.sample_rate:
                    num_samples = int(len(data) * self.sample_rate / fs)
                    data = signal.resample(data, num_samples)
                
                # 归一化
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val
                
                self.audio_data = data
                self.processor.set_audio_data(data)
                
                # 更新按钮状态
                self.update_button_states()
                
                # 更新音频信息
                duration = len(data) / self.sample_rate
                self.audio_info_label.setText(f"音频: {filename} ({duration:.2f}秒)")
                self.statusBar().showMessage(f"音频已加载: {duration:.2f}秒")
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载音频文件失败: {str(e)}")
    
    def browse_hrtf_folder(self):
        """浏览HRTF文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择HRTF文件夹")
        
        if folder_path:
            try:
                folder_name = os.path.basename(folder_path)
                self.hrtf_path_label.setText(folder_name)
                
                # 加载HRTF数据库
                self.hrtf_db = HRTFDatabase(sample_rate=self.sample_rate)
                self.hrtf_db.load_from_folder(folder_path)
                self.processor.set_hrtf_database(self.hrtf_db)
                
                # 更新按钮状态
                self.update_button_states()
                
                # 初始化位置
                self.update_position()
                
                # 更新HRTF信息
                self.hrtf_info_label.setText(f"HRTF: {folder_name} ({len(self.hrtf_db.hrtfs)}个)")
                self.statusBar().showMessage(f"HRTF已加载: {len(self.hrtf_db.hrtfs)}个")
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载HRTF失败: {str(e)}")
    
    def update_position(self):
        """更新位置"""
        slider_azimuth = self.azimuth_slider.value()
        elevation = self.elevation_slider.value()
        
        # 获取映射后的方位角
        mapped_azimuth = self.map_slider_to_hrtf_azimuth(slider_azimuth)
        
        # 更新标签显示
        self.azimuth_label.setText(f"方位角: {slider_azimuth}°")
        self.elevation_label.setText(f"仰角: {elevation}°")
        self.position_info_label.setText(f"当前位置: 方位角 {slider_azimuth}°, 仰角 {elevation}°")
        
        if self.processor and self.hrtf_db:
            active_count = self.processor.update_position(mapped_azimuth, elevation)
            max_hrtfs = int(self.hrtf_limit_combo.currentText())
            self.active_hrtf_label.setText(f"活跃HRTF: {active_count}/{max_hrtfs}个")
    
    def map_slider_to_hrtf_azimuth(self, slider_value):
        """将滑块值映射到HRTF方位角"""
        if slider_value <= 0:
            return -slider_value
        else:
            return 360 - slider_value
    
    def update_volume(self):
        """更新音量"""
        volume_value = self.volume_slider.value()
        volume = volume_value / 100.0
        
        self.volume_label.setText(f"音量: {volume_value}%")
        
        if self.processor:
            self.processor.volume = volume
    
    def update_block_size(self):
        """更新块大小"""
        self.block_size = int(self.block_combo.currentText())
        self.processor.block_size = self.block_size
        print(f"块大小已更新为: {self.block_size}")
        
        # 重新初始化处理器
        if self.hrtf_db:
            self.processor.set_hrtf_database(self.hrtf_db)
            self.update_position()
    
    def update_hrtf_limit(self):
        """更新HRTF数量限制"""
        max_hrtfs = int(self.hrtf_limit_combo.currentText())
        self.processor.max_hrtfs = max_hrtfs
        print(f"HRTF数量限制已更新为: {max_hrtfs}")
        
        # 重新计算位置
        self.update_position()
    
    def update_smooth_transition(self):
        """更新平滑过渡设置"""
        if self.processor:
            self.processor.smooth_transition_enabled = self.smooth_transition_checkbox.isChecked()
            print(f"平滑过渡: {'启用' if self.processor.smooth_transition_enabled else '禁用'}")
    
    def start_playback(self):
        """开始播放"""
        if self.processor:
            success = self.processor.start_playback()
            if success:
                self.statusBar().showMessage("正在播放...")
                self.play_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
            else:
                self.statusBar().showMessage("播放启动失败")
    
    def stop_playback(self):
        """停止播放"""
        if self.processor:
            self.processor.stop_playback()
            self.statusBar().showMessage("播放已停止")
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def test_audio(self):
        """测试音频"""
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.column_stack([test_tone, test_tone])
        
        try:
            sd.play(stereo, self.sample_rate)
            sd.wait()
            self.statusBar().showMessage("测试音播放完成")
        except Exception as e:
            self.statusBar().showMessage(f"测试音播放失败: {e}")
    
    def update_button_states(self):
        """更新按钮状态"""
        audio_loaded = self.audio_data is not None
        hrtf_loaded = self.hrtf_db is not None
        
        self.play_btn.setEnabled(audio_loaded and hrtf_loaded)
    
    def update_progress(self):
        """更新播放进度"""
        try:
            if self.processor.is_playing:
                progress = self.processor.get_progress() * 100
                self.progress_bar.setValue(int(progress))
                
                if progress < 100:
                    self.statusBar().showMessage(f"正在播放... {progress:.1f}%")
                else:
                    self.statusBar().showMessage("播放完成")
                    self.play_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    self.progress_bar.setValue(0)
                    
        except Exception as e:
            print(f"进度更新错误: {e}")
    
    def update_performance(self):
        """更新性能信息"""
        try:
            if self.processor:
                avg_time = self.processor.get_avg_processing_time()
                delay_comp = self.processor.delay_compensation
                delay_ms = delay_comp / self.sample_rate * 1000
                
                self.perf_label.setText(
                    f"处理延迟: {avg_time:.1f}ms | 卷积延迟: {delay_comp}采样点 ({delay_ms:.1f}ms)"
                )
        except Exception as e:
            print(f"性能更新错误: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.processor:
            self.processor.stop_playback()
        
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        
        if self.perf_timer.isActive():
            self.perf_timer.stop()
        
        event.accept()

# ==================== 主函数 ====================
def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()