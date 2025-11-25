// src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod types;
mod vjoy;
mod engine;
mod gui;
mod recorder;
mod visualizer;

use eframe::egui;

// 字体设置函数
fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    
    // 1. 加载自定义中文字体
    // 注意：你必须保证 CJK_Font.ttf 存在于项目根目录！
    let cjk_data = egui::FontData::from_static(include_bytes!("../CJK_Font.ttf")); 
    fonts.font_data.insert("custom_cjk_font".to_owned(), cjk_data);

    // 2. 修复字体堆栈：在默认字体之后，追加 CJK 字体作为备选。
    // 这样基础的英文字符会优先使用 egui 自带的字体。
    if let Some(proportional) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        proportional.push("custom_cjk_font".to_owned());
    }
    if let Some(monospace) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        monospace.push("custom_cjk_font".to_owned());
    }
    
    ctx.set_fonts(fonts);
}

// 入口函数
fn main() -> eframe::Result<()> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_title("QNMDsol demo v0.1"),
        ..Default::default()
    };
    
    eframe::run_native(
        "QNMDsol",
        options,
        Box::new(|cc| {
            // 调用字体设置函数，传入 egui 上下文
            setup_fonts(&cc.egui_ctx); 
            Box::new(gui::QnmdSolApp::default())
        }),
    )
}