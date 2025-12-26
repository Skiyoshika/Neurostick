import time
import os
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def record_session(label, duration=10):
    # 1. 配置板卡 (COM4)
    params = BrainFlowInputParams()
    params.serial_port = "COM4" # ⚠️ 如果连接失败，请检查端口
    
    # Board ID 2 = Cyton + Daisy (16通道)
    board_id = BoardIds.CYTON_DAISY_BOARD.value
    
    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        
        print(f"🔴 开始录制 [{label}] ... 保持 {duration} 秒")
        time.sleep(duration)
        
        # 获取数据并停止
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()
        
        # 2. 保存为 CSV (格式必须与 Rust 保持一致)
        # Rust 格式: Timestamp, Ch0...Ch15
        # BrainFlow 原始数据里: 
        # Row 0 = Package Num (我们不需要)
        # Row 1-16 = EEG Data (我们需要)
        # Row 22 = Timestamp (我们需要)
        
        # 提取 EEG 和 时间戳
        eeg_channels = board.get_eeg_channels(board_id)
        timestamp_channel = board.get_timestamp_channel(board_id)
        
        # 拼接数据: [Timestamp, EEG_1 ... EEG_16]
        # 注意：我们需要转置(Transpose)成 (n_samples, n_features)
        df_data = pd.DataFrame(data[eeg_channels].T) 
        timestamps = data[timestamp_channel].T
        
        # 插入时间戳到第一列
        df_data.insert(0, "Timestamp", timestamps)
        
        # 重命名列头，对齐 Rust 的格式
        cols = ["Timestamp"] + [f"Ch{i}" for i in range(len(eeg_channels))]
        df_data.columns = cols
        
        # 保存文件
        timestamp_str = int(time.time())
        filename = f"../training_data_{label}_{timestamp_str}.csv"
        df_data.to_csv(filename, index=False)
        
        print(f"✅ 数据已保存: {filename}")
        print(f"📊 样本数: {df_data.shape[0]}")

    except Exception as e:
        print(f"❌ 录制失败: {e}")

if __name__ == "__main__":
    print("=== Neurostick Python Recorder (Backup) ===")
    lbl = input("请输入动作标签 (例如 Relax, Attack): ").strip()
    if lbl:
        sec = input("请输入录制时长 (秒, 默认10): ").strip()
        duration = int(sec) if sec.isdigit() else 10
        record_session(lbl, duration)