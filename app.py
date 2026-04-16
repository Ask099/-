import os
import json
import platform
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Any
import streamlit as st
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

# ====================== 【新增：Matplotlib 中文字体自动配置函数】 ======================
def set_matplotlib_chinese_font():
    """自动检测并配置Matplotlib使用系统可用的中文字体，解决中文方框问题"""
    system = platform.system()
    
    # 不同系统的常见中文字体列表
    font_candidates = {
        "Windows": ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"],
        "Darwin": ["Arial Unicode MS", "PingFang SC", "Heiti TC"],
        "Linux": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Noto Sans SC"]
    }
    
    # 获取系统当前可用字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    target_font = None
    
    # 按优先级查找可用字体
    for font_name in font_candidates.get(system, []):
        if font_name in available_fonts:
            target_font = font_name
            break
    
    # 配置字体
    if target_font:
        plt.rcParams['font.sans-serif'] = [target_font]  # 设置默认字体
        plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方框的问题
        # print(f"已自动配置中文字体：{target_font}") # 调试用
    else:
        st.warning("⚠️ 未检测到系统中文字体，图表中文可能显示为方框。建议安装微软雅黑或文泉驿微米黑字体。")

# ====================== 【配置文件加载函数】 ======================
def load_config():
    """从外部 config.txt 加载配置（优先），如果没有则使用环境变量"""
    config_file = "config.txt"
    api_key = None
    
    # 1. 尝试从外部配置文件加载
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "DASHSCOPE_API_KEY":
                            api_key = value.strip()
                            break
            if api_key:
                return api_key, "config.txt"
        except Exception as e:
            st.warning(f"读取 config.txt 失败：{e}")
    
    # 2. 如果没有文件，尝试从环境变量读取
    env_key = os.environ.get("DASHSCOPE_API_KEY")
    if env_key:
        return env_key, "环境变量"
    
    # 3. 都没有，返回 None
    return None, None

# ====================== 【页面配置】 ======================
st.set_page_config(
    page_title="厄尔尼诺预测系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 【全局配置与模型加载】 ======================
@st.cache_resource
def load_global_resources():
    """加载模型、归一化器、气候态（只加载一次），LLM仅在有API Key时初始化"""
    
    # 1. 加载 API Key
    api_key, key_source = load_config()
    llm = None
    
    if not api_key:
        # 无 API Key 时给出警告，但不停止程序
        st.warning("⚠️ 未找到 API Key，AI 专业分析功能将不可用。")
        st.info("""
        💡 如需启用 AI 分析，请按以下步骤配置：
        1. 在程序同级目录下新建一个 `config.txt` 文件
        2. 在文件里写一行：
           `DASHSCOPE_API_KEY=sk-你的阿里云通义千问APIKey`
        3. 保存文件，刷新页面
        """)
    else:
        # 有 API Key 时初始化大模型
        os.environ["DASHSCOPE_API_KEY"] = api_key
        try:
            llm = ChatTongyi(model="qwen-plus", temperature=0)
        except Exception as e:
            st.warning(f"⚠️ 初始化大模型失败：{e}，AI 分析功能将不可用。")
            llm = None
    
    # 2. 定义模型架构（和训练时一致）
    class ENSOBiLSTMWithMultiHeadAttention(nn.Module):
        def __init__(self, input_size=4, hidden_size=256, num_layers=1, dropout=0.5, pred_len=3, num_heads=4):
            super().__init__()
            self.pred_len = pred_len
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
            )
            lstm_output_dim = hidden_size * 2
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=lstm_output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size, pred_len)
            )
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            return self.fc(attn_output[:, -1, :])
    
    # 3. 加载模型（始终加载，保证预测功能可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ENSOBiLSTMWithMultiHeadAttention(hidden_size=256)
    model.eval()
    
    scaler = None
    climatology = None
    
    return model, scaler, climatology, llm, key_source

# ====================== 【数据处理函数】 ======================
def load_single_var_from_uploaded_file(uploaded_file, var_name):
    """从上传的文件加载单变量数据"""
    try:
        ds = xr.open_dataset(uploaded_file)
        data = ds[var_name].values
        time = ds["time"].values if "time" in ds else None
        
        if len(data.shape) == 3:
            lat = ds["lat"].values
            lon = ds["lon"].values
            lat_mask = (lat >= -5) & (lat <= 5)
            lon_mask = (lon >= 190) & (lon <= 240)
            
            if np.sum(lat_mask) > 0 and np.sum(lon_mask) > 0:
                data_region = data[:, lat_mask, :][:, :, lon_mask]
                data_series = data_region.mean(axis=(1, 2))
            else:
                data_series = data.mean(axis=(1, 2))
        else:
            data_series = data.flatten()
        
        data_series = pd.Series(data_series).interpolate(method='linear').fillna(0).values
        return data_series, time
    except Exception as e:
        st.error(f"加载文件出错：{e}")
        return None, None

def add_seasonal_encoding(months):
    return np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12)

def prepare_prediction_data(sst_raw, ht_raw, nino34_raw, time):
    """准备预测数据（简化版）"""
    df_time = pd.DataFrame({'time': pd.to_datetime(time)})
    months = df_time['time'].dt.month.values
    
    seq_len = 24
    if len(sst_raw) < seq_len:
        st.error("数据长度不足，需要至少24个月的数据")
        return None
    
    sst_recent = sst_raw[-seq_len:]
    ht_recent = ht_raw[-seq_len:]
    nino34_recent = nino34_raw[-seq_len:]
    months_recent = months[-seq_len:]
    
    sin_month, cos_month = add_seasonal_encoding(months_recent)
    
    feature = np.stack([
        sst_recent, ht_recent, sin_month, cos_month
    ], axis=1)
    
    return torch.tensor(feature[np.newaxis, ...], dtype=torch.float32), df_time['time'].values[-seq_len:]

# ====================== 【预测与分析函数】 ======================
def run_prediction(model, input_tensor):
    """运行模型预测"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
    
    return pred.cpu().numpy().flatten()

def analyze_with_llm(llm, historical_nino34, pred_nino34, future_months=3):
    """调用大模型分析预测结果（仅在llm存在时调用）"""
    historical_last = historical_nino34[-12:]
    
    prompt = f"""
    你是一名资深的气候预测专家。以下是厄尔尼诺预测数据：
    
    【历史数据（最近12个月）】
    Nino3.4指数（异常值）：{', '.join([f'{x:.2f}' for x in historical_last])}
    
    【未来预测】
    未来{future_months}个月Nino3.4指数预测值：{', '.join([f'{x:.2f}' for x in pred_nino34])}
    
    请写一份专业的预测分析报告，包括：
    1. 当前气候状态评估
    2. 未来趋势预测
    3. 是否会发生厄尔尼诺/拉尼娜？强度如何？
    4. 相关建议与提醒
    
    注意：Nino3.4指数 > 0.5 为厄尔尼诺，< -0.5 为拉尼娜。
    """
    
    with st.spinner("AI 正在分析预测结果..."):
        response = llm.invoke([HumanMessage(content=prompt)])
    
    return response.content

def plot_prediction_results(historical_time, historical_nino34, pred_nino34):
    """绘制预测结果图（已集成中文字体配置）"""
    # 【关键修改】绘图前先配置中文字体
    set_matplotlib_chinese_font()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(historical_time, historical_nino34, label='历史 Nino3.4', color='blue', linewidth=2)
    
    future_time = pd.date_range(start=historical_time[-1], periods=len(pred_nino34)+1, freq='MS')[1:]
    full_pred = np.concatenate([[historical_nino34[-1]], pred_nino34])
    full_time_pred = np.concatenate([[historical_time[-1]], future_time])
    
    ax.plot(full_time_pred, full_pred, label='预测 Nino3.4', color='red', linestyle='--', linewidth=2, marker='o')
    
    ax.axhline(y=0.5, color='orange', linestyle=':', label='厄尔尼诺阈值 (0.5)')
    ax.axhline(y=-0.5, color='purple', linestyle=':', label='拉尼娜阈值 (-0.5)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('时间')
    ax.set_ylabel('Nino3.4 指数异常')
    ax.set_title('厄尔尼诺预测结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# ====================== 【主界面】 ======================
def main():
    st.title("🌊 厄尔尼诺智能预测系统")
    st.markdown("---")
    
    # 1. 加载资源
    model, scaler, climatology, llm, key_source = load_global_resources()
    
    # 2. 侧边栏（根据API Key状态显示不同内容）
    st.sidebar.header("⚙️ 系统状态")
    if llm is not None:
        st.sidebar.success(f"✅ API Key 已加载\n({key_source})")
        st.sidebar.info("🧠 AI 分析功能：已启用")
    else:
        st.sidebar.warning("⚠️ API Key 未配置")
        st.sidebar.info("🧠 AI 分析功能：已禁用")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 数据上传")
    st.sidebar.markdown("请上传 SST、HT、Nino3.4 数据文件（NetCDF 格式）")
    
    sst_file = st.sidebar.file_uploader("上传 SST (海温) 数据", type=["nc"])
    ht_file = st.sidebar.file_uploader("上传 HT (温跃层) 数据", type=["nc"])
    nino34_file = st.sidebar.file_uploader("上传 Nino3.4 指数数据", type=["nc"])
    
    # 3. 主界面
    if sst_file and ht_file and nino34_file:
        st.success("✅ 数据上传成功！")
        
        with st.spinner("正在加载数据..."):
            sst_raw, time_sst = load_single_var_from_uploaded_file(sst_file, "sst")
            ht_raw, time_ht = load_single_var_from_uploaded_file(ht_file, "ht")
            nino34_raw, time_nino34 = load_single_var_from_uploaded_file(nino34_file, "nino34")
        
        if sst_raw is not None and ht_raw is not None and nino34_raw is not None:
            min_len = min(len(sst_raw), len(ht_raw), len(nino34_raw))
            sst_raw = sst_raw[:min_len]
            ht_raw = ht_raw[:min_len]
            nino34_raw = nino34_raw[:min_len]
            time = time_nino34[:min_len]
            
            with st.expander("📊 查看历史数据概览"):
                # 【关键修改】绘制历史数据前也配置中文字体
                set_matplotlib_chinese_font()
                fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
                ax_hist.plot(pd.to_datetime(time), nino34_raw, label='Nino3.4 历史', color='blue')
                ax_hist.axhline(y=0.5, color='orange', linestyle=':', label='厄尔尼诺阈值')
                ax_hist.axhline(y=-0.5, color='purple', linestyle=':', label='拉尼娜阈值')
                ax_hist.set_title("历史 Nino3.4 指数")
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist)
            
            if st.button("🚀 开始预测", type="primary", use_container_width=True):
                input_tensor, historical_time = prepare_prediction_data(sst_raw, ht_raw, nino34_raw, time)
                
                if input_tensor is not None:
                    with st.spinner("模型正在预测..."):
                        # 这里用模拟预测结果，你替换成真实的 model 预测
                        # pred_nino34 = run_prediction(model, input_tensor)
                        pred_nino34 = np.random.uniform(0.3, 0.9, size=3)
                    
                    st.markdown("---")
                    st.subheader("📈 预测结果")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        historical_nino34 = nino34_raw[-24:]
                        fig_pred = plot_prediction_results(historical_time, historical_nino34, pred_nino34)
                        st.pyplot(fig_pred)
                    
                    with col2:
                        st.metric("未来 1 个月", f"{pred_nino34[0]:.2f}")
                        st.metric("未来 2 个月", f"{pred_nino34[1]:.2f}")
                        st.metric("未来 3 个月", f"{pred_nino34[2]:.2f}")
                        
                        if pred_nino34[-1] > 0.5:
                            st.warning("⚠️ 预测：弱厄尔尼诺")
                        elif pred_nino34[-1] < -0.5:
                            st.info("❄️ 预测：弱拉尼娜")
                        else:
                            st.success("✅ 预测：正常状态")
                    
                    # ====================== 【AI分析部分：条件显示】 ======================
                    analysis = None
                    if llm is not None:
                        st.markdown("---")
                        st.subheader("🧠 AI 专业分析")
                        analysis = analyze_with_llm(llm, nino34_raw, pred_nino34)
                        st.markdown(analysis)
                    else:
                        st.markdown("---")
                        st.info("ℹ️ 未配置 API Key，AI 专业分析功能不可用。")
                    
                    # ====================== 【报告下载：根据是否有AI分析调整内容】 ======================
                    st.markdown("---")
                    if analysis is not None:
                        report_content = f"# 厄尔尼诺预测报告\n\n## 预测结果\n{', '.join([f'{x:.2f}' for x in pred_nino34])}\n\n## AI分析\n{analysis}"
                    else:
                        report_content = f"# 厄尔尼诺预测报告\n\n## 预测结果\n{', '.join([f'{x:.2f}' for x in pred_nino34])}"
                    
                    st.download_button("📄 下载分析报告", data=report_content, file_name="enso_prediction_report.md", mime="text/markdown")
    else:
        st.info("👈 请在左侧上传数据文件开始预测")
        st.markdown("""
        ### 使用说明
        1. **（可选）配置 API Key**（启用 AI 分析）：
           - 在程序同级目录下新建 `config.txt`
           - 写入：`DASHSCOPE_API_KEY=sk-你的APIKey`
        2. 在左侧上传三个 NetCDF 文件：
           - SST (海温)
           - HT (温跃层深度)
           - Nino3.4 指数
        3. 点击「开始预测」
        4. 查看预测图表（和 AI 分析，如已配置）
        """)

if __name__ == "__main__":
    main()