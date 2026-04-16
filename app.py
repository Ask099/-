import os
import sys
import subprocess
import time
import socket

def get_free_port():
    """【方案3核心】自动获取一个可用的端口"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    # ====================== 1. 路径处理 ======================
    if getattr(sys, 'frozen', False):
        # 打包后的环境
        base_path = sys._MEIPASS
        sys.path.insert(0, base_path)
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 找到 app.py
    app_path = os.path.join(base_path, "Project.py")
    if not os.path.exists(app_path):
        print(f"❌ 错误：找不到 app.py，路径：{app_path}")
        safe_exit(1)
    
    # ====================== 2. 找到 Streamlit ======================
    streamlit_exe = None
    
    if getattr(sys, 'frozen', False):
        possible_paths = [
            os.path.join(base_path, "streamlit.exe"),
            os.path.join(base_path, "Scripts", "streamlit.exe"),
            os.path.join(os.path.dirname(sys.executable), "streamlit.exe"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                streamlit_exe = p
                break
    
    if streamlit_exe is None:
        streamlit_exe = "streamlit"
    
    # ====================== 3. 【方案3】自动获取可用端口 ======================
    port = get_free_port()
    
    print("="*60)
    print("🌊 厄尔尼诺智能预测系统")
    print("="*60)
    print(f"📂 基础路径: {base_path}")
    print(f"📄 应用入口: {app_path}")
    print(f"🚀 Streamlit: {streamlit_exe}")
    print(f"🎯 自动选择端口: {port}")  # 显示自动选的端口
    print("="*60)
    print("\n⏳ 正在启动服务...")
    print(f"💡 提示：请稍等 5-10 秒，浏览器将自动打开")
    print(f"💡 如果浏览器未打开，请手动访问：http://localhost:{port}")
    print("="*60)
    
    # ====================== 4. 自动打开浏览器（用动态端口） ======================
    def open_browser():
        try:
            import webbrowser
            time.sleep(5)
            webbrowser.open(f"http://localhost:{port}")  # 用动态端口
        except Exception as e:
            print(f"⚠️  自动打开浏览器失败：{e}")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # ====================== 5. 启动 Streamlit（用动态端口） ======================
    cmd = [
        streamlit_exe,
        "run",
        app_path,
        "--server.headless", "true",
        "--server.port", str(port),  # 【关键】用动态端口
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        print("\n📜 服务日志：")
        print("-" * 60)
        for line in process.stdout:
            print(line, end='')
            
    except KeyboardInterrupt:
        print("\n👋 收到退出信号")
    except Exception as e:
        print(f"\n❌ 启动错误：{e}")
        print(f"\n💡 调试信息：")
        print(f"   命令：{' '.join(cmd)}")
        print(f"   当前目录：{os.getcwd()}")
        print(f"   PATH：{os.environ.get('PATH', '')}")
    finally:
        safe_exit(0)

def safe_exit(code=0):
    print("\n" + "="*60)
    print("👋 系统已退出")
    print("="*60)
    
    if getattr(sys, 'frozen', False) and not sys.stdout:
        time.sleep(3)
    
    sys.exit(code)

if __name__ == "__main__":
    main()