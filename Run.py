#!/usr/bin/env python3
"""
简化运行脚本 - 修复版
"""
import subprocess
import sys
import os

def install_requirements():
    """安装依赖"""
    print("正在安装依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖安装完成！")
    except subprocess.CalledProcessError as e:
        print(f"安装依赖失败: {e}")
        return False
    return True

def check_requirements():
    """检查依赖"""
    try:
        import numpy
        import scipy
        import sounddevice
        import soundfile
        import matplotlib
        import PyQt5
        print("所有依赖已安装！")
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        return False

def main():
    """主函数"""
    print("=== 实时音频空间化系统 ===")
    
    # 检查依赖
    if not check_requirements():
        print("正在尝试自动安装依赖...")
        if not install_requirements():
            print("请手动安装依赖:")
            print("pip install -r requirements.txt")
            input("按Enter键退出...")
            return
    
    # 首先检查文件是否存在
    target_file = "SoundSpatialSimpleRealtime.py"
    if not os.path.exists(target_file):
        print(f"错误：找不到文件 {target_file}")
        print(f"当前目录：{os.getcwd()}")
        print(f"目录中的文件：")
        for f in os.listdir('.'):
            print(f"  - {f}")
        input("按Enter键退出...")
        return
    
    # 尝试不同的导入方式
    try:
        # 方法1：直接导入模块
        print(f"尝试导入 {target_file}...")
        import importlib.util
        spec = importlib.util.spec_from_file_location("SoundSpatialSimpleRealtime", target_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找并运行主函数
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, 'SoundSpatialSimpleRealtime'):
            module.SoundSpatialSimpleRealtime()
        else:
            # 尝试运行模块本身
            import runpy
            runpy.run_path(target_file, run_name="__main__")
            
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试直接执行文件
        print("\n尝试直接执行脚本...")
        try:
            subprocess.run([sys.executable, target_file])
        except Exception as e2:
            print(f"直接执行也失败: {e2}")
        
        input("按Enter键退出...")

if __name__ == "__main__":
    main()