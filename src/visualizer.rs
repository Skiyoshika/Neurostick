// src/visualizer.rs
use crate::types::GamepadState;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Rounding, Shape, Stroke, Vec2};

/// 绘制写实风格的 Xbox 手柄 (包含正面和顶面视图)
pub fn draw_xbox_controller(ui: &mut egui::Ui, gamepad: &GamepadState) {
    // === 配色方案 ===
    let body_color = Color32::from_rgb(50, 50, 55);
    let outline_color = Color32::from_rgb(80, 80, 85);
    let btn_base_color = Color32::from_rgb(70, 70, 75);
    let highlight_color = Color32::from_rgb(200, 200, 200);
    let text_color = Color32::from_rgb(180, 180, 180);

    let width = 280.0;
    let height_front = 180.0;
    let height_back = 60.0;
    let spacing = 15.0;
    let total_height = height_front + height_back + spacing;

    let (response, painter) = ui.allocate_painter(Vec2::new(width, total_height), egui::Sense::hover());
    let rect = response.rect;
    let top_left = rect.min;

    // ----------------------------------------------------------------
    // 1. 顶视图 (Top View) - 展示 LB/RB/LT/RT
    // ----------------------------------------------------------------
    let top_view_rect = Rect::from_min_size(top_left, Vec2::new(width, height_back));
    painter.text(top_view_rect.min + Vec2::new(5.0, 0.0), egui::Align2::LEFT_TOP, "TOP VIEW", egui::FontId::proportional(10.0), text_color);

    let top_body_rect = top_view_rect.shrink2(Vec2::new(20.0, 10.0)).translate(Vec2::new(0.0, 5.0));
    painter.rect_filled(top_body_rect, Rounding::same(8.0), body_color);
    painter.rect_stroke(top_body_rect, Rounding::same(8.0), Stroke::new(1.5, outline_color));

    // 绘制扳机 (LT/RT)
    let trigger_size = Vec2::new(45.0, 20.0);
    let lt_pos = top_body_rect.left_center() + Vec2::new(trigger_size.x / 2.0 - 5.0, 0.0);
    let rt_pos = top_body_rect.right_center() - Vec2::new(trigger_size.x / 2.0 - 5.0, 0.0);

    let draw_trigger = |center: Pos2, active: bool, label: &str| {
        let r = Rect::from_center_size(center, trigger_size);
        let fill = if active { Color32::from_rgb(200, 50, 50) } else { btn_base_color };
        painter.rect_filled(r, Rounding::same(4.0), fill);
        painter.rect_stroke(r, Rounding::same(4.0), Stroke::new(1.0, outline_color));
        painter.text(center, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(12.0), if active { Color32::WHITE } else { text_color });
    };
    draw_trigger(lt_pos, gamepad.lt, "LT");
    draw_trigger(rt_pos, gamepad.rt, "RT");

    // 绘制肩键 (LB/RB)
    let bumper_size = Vec2::new(40.0, 14.0);
    let lb_pos = lt_pos + Vec2::new(trigger_size.x / 2.0 + bumper_size.x / 2.0 + 2.0, 0.0);
    let rb_pos = rt_pos - Vec2::new(trigger_size.x / 2.0 + bumper_size.x / 2.0 + 2.0, 0.0);

    let draw_bumper = |center: Pos2, active: bool, label: &str| {
        let r = Rect::from_center_size(center, bumper_size);
        let fill = if active { Color32::from_rgb(50, 200, 200) } else { btn_base_color };
        painter.rect_filled(r, Rounding::same(2.0), fill);
        painter.rect_stroke(r, Rounding::same(2.0), Stroke::new(1.0, outline_color));
        painter.text(center, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(10.0), if active { Color32::BLACK } else { text_color });
    };
    draw_bumper(lb_pos, gamepad.lb, "LB");
    draw_bumper(rb_pos, gamepad.rb, "RB");

    // ----------------------------------------------------------------
    // 2. 正视图 (Face View) - 展示摇杆和 ABXY
    // ----------------------------------------------------------------
    let face_rect = Rect::from_min_size(top_left + Vec2::new(0.0, height_back + spacing), Vec2::new(width, height_front));
    painter.text(face_rect.min + Vec2::new(5.0, 0.0), egui::Align2::LEFT_TOP, "FACE VIEW", egui::FontId::proportional(10.0), text_color);
    let fc = face_rect.center();

    // 手柄轮廓
    let body_points = vec![
        fc + Vec2::new(-70.0, -40.0), fc + Vec2::new(70.0, -40.0), 
        fc + Vec2::new(110.0, 20.0),  fc + Vec2::new(70.0, 60.0),  
        fc + Vec2::new(-70.0, 60.0), fc + Vec2::new(-110.0, 20.0), 
    ];
    painter.add(Shape::convex_polygon(body_points.clone(), body_color, Stroke::new(1.5, outline_color)));

    // 左摇杆
    let ls_c = fc + Vec2::new(-65.0, -10.0);
    painter.circle_filled(ls_c, 28.0, btn_base_color);
    painter.circle_stroke(ls_c, 28.0, Stroke::new(1.0, outline_color));
    let ls_head = ls_c + Vec2::new(gamepad.lx, -gamepad.ly) * 12.0;
    let ls_act = gamepad.lx.abs() > 0.1 || gamepad.ly.abs() > 0.1;
    painter.circle_filled(ls_head, 16.0, body_color);
    painter.circle_stroke(ls_head, 16.0, Stroke::new(2.0, if ls_act { Color32::from_rgb(0, 255, 255) } else { outline_color }));
    painter.circle_filled(ls_head, 12.0, if ls_act { Color32::from_rgb(0, 100, 100) } else { btn_base_color });

    // 右摇杆
    let rs_c = fc + Vec2::new(40.0, 35.0);
    painter.circle_filled(rs_c, 28.0, btn_base_color);
    painter.circle_stroke(rs_c, 28.0, Stroke::new(1.0, outline_color));
    let rs_head = rs_c + Vec2::new(gamepad.rx, -gamepad.ry) * 12.0;
    let rs_act = gamepad.rx.abs() > 0.1 || gamepad.ry.abs() > 0.1;
    painter.circle_filled(rs_head, 16.0, body_color);
    painter.circle_stroke(rs_head, 16.0, Stroke::new(2.0, if rs_act { Color32::from_rgb(255, 0, 255) } else { outline_color }));
    painter.circle_filled(rs_head, 12.0, if rs_act { Color32::from_rgb(100, 0, 100) } else { btn_base_color });

    // D-Pad
    let dpad_c = fc + Vec2::new(-40.0, 35.0);
    let d_sz = 10.0;
    let draw_dpad_arm = |offset: Vec2, active: bool| {
        let r = Rect::from_center_size(dpad_c + offset, Vec2::splat(d_sz));
        let c = if active { Color32::from_rgb(255, 165, 0) } else { btn_base_color };
        painter.rect_filled(r, Rounding::same(2.0), c);
        painter.rect_stroke(r, Rounding::same(2.0), Stroke::new(1.0, outline_color));
    };
    draw_dpad_arm(Vec2::new(0.0, 0.0), false);
    draw_dpad_arm(Vec2::new(0.0, -d_sz), gamepad.dpad_up);
    draw_dpad_arm(Vec2::new(0.0, d_sz), gamepad.dpad_down);
    draw_dpad_arm(Vec2::new(-d_sz, 0.0), gamepad.dpad_left);
    draw_dpad_arm(Vec2::new(d_sz, 0.0), gamepad.dpad_right);

    // ABXY
    let btn_c = fc + Vec2::new(70.0, -30.0);
    let b_rad = 11.0;
    let b_gap = 20.0;
    let draw_face_btn = |offset: Vec2, active: bool, label: &str, color: Color32| {
        let pos = btn_c + offset;
        let fill = if active { color } else { btn_base_color };
        painter.circle_filled(pos, b_rad, fill);
        painter.circle_stroke(pos, b_rad, Stroke::new(1.0, outline_color));
        painter.text(pos, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(14.0), if active { Color32::BLACK } else { color });
    };
    draw_face_btn(Vec2::new(0.0, b_gap), gamepad.a, "A", Color32::GREEN);
    draw_face_btn(Vec2::new(b_gap, 0.0), gamepad.b, "B", Color32::RED);
    draw_face_btn(Vec2::new(-b_gap, 0.0), gamepad.x, "X", Color32::BLUE);
    draw_face_btn(Vec2::new(0.0, -b_gap), gamepad.y, "Y", Color32::YELLOW);

    // Start / Back
    let draw_small_btn = |center: Pos2, label: &str| {
        painter.circle_filled(center, 7.0, btn_base_color);
        painter.circle_stroke(center, 7.0, Stroke::new(1.0, outline_color));
        painter.text(center, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(8.0), text_color);
    };
    draw_small_btn(fc + Vec2::new(-20.0, -10.0), "<");
    draw_small_btn(fc + Vec2::new(20.0, -10.0), ">");
}