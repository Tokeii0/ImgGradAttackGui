import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageQt
import importlib.util
import inspect
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox,
    QLineEdit, QProgressBar, QMessageBox, QGroupBox, QFormLayout,
    QTextEdit, QComboBox, QSplitter, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon

class ModelRecoveryThread(QThread):
    """线程类，用于后台执行梯度优化过程"""
    progress_signal = Signal(int, float)  # 进度信号：(步骤, 当前输出值)
    image_signal = Signal(object)  # 图像信号：发送PIL图像
    finished_signal = Signal(bool, str)  # 完成信号：(成功/失败, 消息)
    
    def __init__(self, model, input_shape, steps, learning_rate, device, use_normalization=False, 
                 output_class_idx=None, pixel_range=None):
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.steps = steps
        self.learning_rate = learning_rate
        self.device = device
        self.running = True
        
        # 新增参数
        self.use_normalization = use_normalization  # 是否使用归一化
        self.output_class_idx = output_class_idx  # 多分类时要关注的类别索引
        self.pixel_range = pixel_range or ((-1, 1) if use_normalization else (0, 1))  # 像素值范围
        
    def run(self):
        try:
            # 将模型移至指定设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 随机初始化输入张量
            min_val, max_val = self.pixel_range
            if min_val < 0:  # 归一化范围如[-1,1]
                x = torch.randn(self.input_shape, requires_grad=True, device=self.device)
            else:  # 标准范围如[0,1]
                x = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            
            # 设置优化器
            optimizer = torch.optim.Adam([x], lr=self.learning_rate)
            
            # 优化循环
            update_interval = max(1, self.steps // 100)  # 更新频率
            
            for i in range(self.steps):
                if not self.running:
                    break
                    
                optimizer.zero_grad()
                out = self.model(x)
                
                # 处理模型输出
                if isinstance(out, torch.Tensor):
                    if out.shape[-1] > 1 and self.output_class_idx is not None:
                        # 多分类情况，取指定类别的概率
                        target_out = out[..., self.output_class_idx]
                    else:
                        # 单输出情况
                        target_out = out.view(-1)[0] if out.numel() > 1 else out
                else:
                    target_out = out
                
                loss = -target_out  # 最大化输出
                loss.backward()
                optimizer.step()
                
                # 限制像素值在指定范围内
                x.data.clamp_(min_val, max_val)
                
                # 发送进度信号
                if i % update_interval == 0 or i == self.steps - 1:
                    self.progress_signal.emit(i, target_out.item())
                    
                    # 发送图像预览
                    if len(self.input_shape) >= 3:  # 确保是图像格式
                        # 如果是归一化的，需要反归一化处理
                        preview_x = x.detach().cpu()
                        if self.use_normalization:
                            # 从[-1,1]映射回[0,1]
                            preview_x = preview_x * 0.5 + 0.5
                        
                        img = self.tensor_to_pil(preview_x)
                        self.image_signal.emit(img)
            
            # 发送最终图像
            final_x = x.detach().cpu()
            if self.use_normalization:
                # 从[-1,1]映射回[0,1]
                final_x = final_x * 0.5 + 0.5
                
            final_img = self.tensor_to_pil(final_x)
            self.image_signal.emit(final_img)
            
            # 保存恢复的张量供后续使用
            self.recovered_tensor = x.detach().cpu()
            self.normalized_tensor = final_x  # 保存反归一化后的张量
            
            self.finished_signal.emit(True, "优化完成！")
            
        except Exception as e:
            error_msg = f"优化过程出错: {str(e)}\n{traceback.format_exc()}"
            self.finished_signal.emit(False, error_msg)
    
    def stop(self):
        self.running = False
    
    def tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        if len(tensor.shape) == 4:  # 批次维度
            tensor = tensor.squeeze(0)
            
        if tensor.shape[0] == 1:  # 单通道
            img = Image.fromarray(
                (tensor.squeeze(0).numpy() * 255).astype(np.uint8), 
                mode='L'
            )
        elif tensor.shape[0] == 3:  # RGB
            img = Image.fromarray(
                (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                mode='RGB'
            )
        else:
            # 其他情况，尝试转换为灰度图
            img = Image.fromarray(
                (tensor[0].numpy() * 255).astype(np.uint8),
                mode='L'
            )
        
        return img

class ModelRecoveryTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.recovery_thread = None
        self.recovered_tensor = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("模型输入恢复工具")
        self.setMinimumSize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(350)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        
        # 模型文件选择
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("选择模型文件(.py)")
        self.select_model_btn = QPushButton("选择模型")
        self.select_model_btn.clicked.connect(self.select_model_file)
        model_layout.addRow(self.select_model_btn, self.model_path_edit)
        
        # 模型类名选择
        self.model_class_combo = QComboBox()
        self.model_class_combo.setEnabled(False)
        model_layout.addRow("模型类名:", self.model_class_combo)
        
        # 权重文件选择
        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setReadOnly(True)
        self.weights_path_edit.setPlaceholderText("选择权重文件(.pth)")
        self.select_weights_btn = QPushButton("加载权重")
        self.select_weights_btn.clicked.connect(self.select_weights_file)
        model_layout.addRow(self.select_weights_btn, self.weights_path_edit)
        
        control_layout.addWidget(model_group)
        
        # 输入设置组
        input_group = QGroupBox("输入设置")
        input_layout = QFormLayout(input_group)
        
        # 输入通道数
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 4)
        self.channels_spin.setValue(1)
        input_layout.addRow("通道数:", self.channels_spin)
        
        # 输入高度
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 1024)
        self.height_spin.setValue(31)
        input_layout.addRow("高度:", self.height_spin)
        
        # 输入宽度
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 1024)
        self.width_spin.setValue(31)
        input_layout.addRow("宽度:", self.width_spin)
        
        control_layout.addWidget(input_group)
        
        # 优化设置组
        optim_group = QGroupBox("优化设置")
        optim_layout = QFormLayout(optim_group)
        
        # 优化步数
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 100000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(1000)
        optim_layout.addRow("优化步数:", self.steps_spin)
        
        # 学习率
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.1)
        optim_layout.addRow("学习率:", self.lr_spin)
        
        # 归一化选项
        self.norm_check = QCheckBox("使用归一化")
        self.norm_check.setToolTip("如果模型训练时使用了Normalize处理，请勾选此项")
        optim_layout.addRow("归一化:", self.norm_check)
        
        # 输出类别选项
        output_layout = QHBoxLayout()
        self.multi_class_check = QCheckBox("多分类输出")
        self.multi_class_check.setToolTip("如果模型输出是多分类结果，请勾选此项")
        self.class_idx_spin = QSpinBox()
        self.class_idx_spin.setRange(0, 100)
        self.class_idx_spin.setValue(1)  # 默认使用第二个类别（索引1）
        self.class_idx_spin.setEnabled(False)
        self.multi_class_check.toggled.connect(self.class_idx_spin.setEnabled)
        output_layout.addWidget(self.multi_class_check)
        output_layout.addWidget(self.class_idx_spin)
        optim_layout.addRow("输出类别:", output_layout)
        
        # 设备选择
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = f"CUDA:{i} ({torch.cuda.get_device_name(i)})"
                self.device_combo.addItem(device_name, f"cuda:{i}")
        optim_layout.addRow("计算设备:", self.device_combo)
        
        control_layout.addWidget(optim_group)
        
        # 操作按钮
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout(actions_group)
        
        self.start_btn = QPushButton("开始恢复")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_recovery)
        actions_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recovery)
        actions_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("保存图片")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_image)
        actions_layout.addWidget(self.save_btn)
        
        control_layout.addWidget(actions_group)
        
        # 进度条
        self.progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        control_layout.addWidget(self.progress_group)
        
        # 添加弹性空间
        control_layout.addStretch()
        
        # 创建右侧预览面板
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        
        # 图像预览标签
        preview_label = QLabel("图像预览")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.image_label.setText("图像将在这里显示")
        preview_layout.addWidget(self.image_label)
        
        # 输出日志
        log_label = QLabel("日志输出")
        preview_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        preview_layout.addWidget(self.log_text)
        
        # 将控制面板和预览面板添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(preview_panel, 1)  # 预览面板可以拉伸
        
        self.log("欢迎使用模型输入恢复工具！请选择模型文件和权重文件开始。")
    
    def select_model_file(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "Python文件 (*.py)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
            self.log(f"已选择模型文件: {file_path}")
            
            # 尝试加载模型类
            try:
                self.load_model_classes(file_path)
            except Exception as e:
                self.show_error(f"加载模型文件出错: {str(e)}")
    
    def load_model_classes(self, file_path):
        """从Python文件中加载模型类"""
        # 获取模块名
        module_name = os.path.basename(file_path).replace('.py', '')
        
        # 加载模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找所有继承自nn.Module的类
        self.model_class_combo.clear()
        self.model_classes = {}
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
                self.model_classes[name] = obj
                self.model_class_combo.addItem(name)
        
        if self.model_classes:
            self.model_class_combo.setEnabled(True)
            self.log(f"找到 {len(self.model_classes)} 个模型类")
        else:
            self.model_class_combo.setEnabled(False)
            self.log("未找到任何继承自nn.Module的类")
    
    def select_weights_file(self):
        """选择权重文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择权重文件", "", "PyTorch权重文件 (*.pth *.pt)"
        )
        
        if file_path:
            self.weights_path_edit.setText(file_path)
            self.log(f"已选择权重文件: {file_path}")
            
            # 检查是否可以开始恢复
            self.check_ready()
    
    def check_ready(self):
        """检查是否可以开始恢复"""
        model_ready = bool(self.model_path_edit.text() and 
                      self.model_class_combo.currentText())
        weights_ready = bool(self.weights_path_edit.text())
        
        self.start_btn.setEnabled(model_ready and weights_ready)
    
    def start_recovery(self):
        """开始恢复过程"""
        try:
            # 获取输入参数
            model_class_name = self.model_class_combo.currentText()
            weights_path = self.weights_path_edit.text()
            
            if not model_class_name or not weights_path:
                self.show_error("请先选择模型类和权重文件")
                return
            
            # 获取输入形状
            channels = self.channels_spin.value()
            height = self.height_spin.value()
            width = self.width_spin.value()
            input_shape = (1, channels, height, width)  # 批次大小为1
            
            # 获取优化参数
            steps = self.steps_spin.value()
            learning_rate = self.lr_spin.value()
            device = self.device_combo.currentData()
            
            # 创建模型实例
            model_class = self.model_classes[model_class_name]
            model = model_class()
            
            # 加载权重
            self.log(f"正在加载权重: {weights_path}")
            try:
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                self.log("权重加载成功")
            except Exception as e:
                self.show_error(f"加载权重失败: {str(e)}")
                return
            
            # 获取归一化和多分类设置
            use_normalization = self.norm_check.isChecked()
            output_class_idx = None
            if self.multi_class_check.isChecked():
                output_class_idx = self.class_idx_spin.value()
                
            # 创建并启动恢复线程
            self.recovery_thread = ModelRecoveryThread(
                model, input_shape, steps, learning_rate, device,
                use_normalization=use_normalization,
                output_class_idx=output_class_idx
            )
            
            # 连接信号
            self.recovery_thread.progress_signal.connect(self.update_progress)
            self.recovery_thread.image_signal.connect(self.update_image)
            self.recovery_thread.finished_signal.connect(self.recovery_finished)
            
            # 更新UI状态
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("正在优化...")
            
            # 启动线程
            self.recovery_thread.start()
            
            self.log(f"开始恢复过程，形状={input_shape}，步数={steps}，学习率={learning_rate}，设备={device}")
            
        except Exception as e:
            self.show_error(f"启动恢复过程失败: {str(e)}\n{traceback.format_exc()}")
    
    def stop_recovery(self):
        """停止恢复过程"""
        if self.recovery_thread and self.recovery_thread.isRunning():
            self.recovery_thread.stop()
            self.log("正在停止恢复过程...")
            self.status_label.setText("正在停止...")
    
    def update_progress(self, step, output_value):
        """更新进度信息"""
        total_steps = self.steps_spin.value()
        progress = int(step / total_steps * 100)
        self.progress_bar.setValue(progress)
        
        self.status_label.setText(f"步骤 {step}/{total_steps}, 输出值: {output_value:.6f}")
        self.log(f"步骤 {step}, 输出值: {output_value:.6f}")
    
    def update_image(self, pil_image):
        """更新图像预览"""
        if pil_image:
            # 转换PIL图像为QPixmap
            qt_image = ImageQt.ImageQt(pil_image)
            pixmap = QPixmap.fromImage(qt_image)
            
            # 调整大小以适应标签
            pixmap = pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # 显示图像
            self.image_label.setPixmap(pixmap)
            
            # 保存最新的恢复图像
            self.recovered_image = pil_image
    
    def recovery_finished(self, success, message):
        """恢复过程完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.status_label.setText("恢复完成")
            self.save_btn.setEnabled(True)
            
            # 保存恢复的张量
            if hasattr(self.recovery_thread, 'recovered_tensor'):
                self.recovered_tensor = self.recovery_thread.recovered_tensor
        else:
            self.status_label.setText("恢复失败")
            self.show_error(message)
        
        self.log(message)
    
    def save_image(self):
        """保存恢复的图像"""
        if hasattr(self, 'recovered_image') and self.recovered_image:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
            )
            
            if file_path:
                try:
                    self.recovered_image.save(file_path)
                    self.log(f"图像已保存至: {file_path}")
                except Exception as e:
                    self.show_error(f"保存图像失败: {str(e)}")
        else:
            self.show_error("没有可保存的图像")
    
    def log(self, message):
        """添加日志消息"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def show_error(self, message):
        """显示错误对话框"""
        QMessageBox.critical(self, "错误", message)
        self.log(f"错误: {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    window = ModelRecoveryTool()
    window.show()
    
    sys.exit(app.exec())
