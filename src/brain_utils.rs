// src/brain_utils.rs
use std::collections::VecDeque;

/// 滑动窗口缓冲区，用于计算脑波能量
pub struct WindowBuffer {
    buffer: VecDeque<f64>,
    capacity: usize,
}

impl WindowBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(size),
            capacity: size,
        }
    }

    /// 推入新数据，保持窗口大小固定
    pub fn push(&mut self, val: f64) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(val);
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }

    /// 计算窗口内的“对数能量” (Log-Power)
    /// 这是 EEG 特征提取的标准方法，比直接看幅度稳得多
    pub fn band_power(&self) -> f64 {
        if self.buffer.is_empty() { return 0.0; }
        
        // 1. 计算均值 (移除残留直流)
        let sum: f64 = self.buffer.iter().sum();
        let mean = sum / self.buffer.len() as f64;

        // 2. 计算方差 (Variance) = 能量 (Power)
        let mut sum_sq = 0.0;
        for &v in self.buffer.iter() {
            let diff = v - mean;
            sum_sq += diff * diff;
        }
        let variance = sum_sq / self.buffer.len() as f64;
        
        // 3. 取对数 (让数据分布更线性，方便阈值判定)
        // 加 1e-6 是为了防止 log(0)
        (variance + 1e-6).ln()
    }
}