// src/visualizer.rs
use crate::types::GamepadState;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Rounding, Shape, Stroke, Vec2};

pub fn draw_xbox_controller(ui: &mut egui::Ui, gamepad: &GamepadState) {
    let body_color = Color32::from_rgb(50, 50, 55);
    let outline_color = Color32::from_rgb(80, 80, 85);
    let btn_base_color = Color32::from_rgb(70, 70, 75);
    let text_color = Color32::from_rgb(180, 180, 180);

    let width = 280.0;
    let height_front = 180.0;
    let height_back = 60.0;
    let spacing = 15.0;
    let total_height = height_front + height_back + spacing;

    let (response, painter) =
        ui.allocate_painter(Vec2::new(width, total_height), egui::Sense::hover());
    let rect = response.rect;
    let top_left = rect.min;

    // 1. Top View
    let top_rect = Rect::from_min_size(top_left, Vec2::new(width, height_back));
    painter.text(
        top_rect.min + Vec2::new(5.0, 0.0),
        egui::Align2::LEFT_TOP,
        "TOP VIEW",
        egui::FontId::proportional(10.0),
        text_color,
    );

    let top_body_rect = top_rect
        .shrink2(Vec2::new(20.0, 10.0))
        .translate(Vec2::new(0.0, 5.0));
    painter.rect_filled(top_body_rect, Rounding::same(8.0), body_color);
    painter.rect_stroke(
        top_body_rect,
        Rounding::same(8.0),
        Stroke::new(1.5, outline_color),
    );

    let trigger_size = Vec2::new(45.0, 20.0);
    let lt_pos = top_body_rect.left_center() + Vec2::new(trigger_size.x / 2.0 - 5.0, 0.0);
    let rt_pos = top_body_rect.right_center() - Vec2::new(trigger_size.x / 2.0 - 5.0, 0.0);

    let draw_trigger = |center: Pos2, active: bool, label: &str| {
        let r = Rect::from_center_size(center, trigger_size);
        let fill = if active {
            Color32::from_rgb(200, 50, 50)
        } else {
            btn_base_color
        };
        painter.rect_filled(r, Rounding::same(4.0), fill);
        painter.rect_stroke(r, Rounding::same(4.0), Stroke::new(1.0, outline_color));
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(12.0),
            if active { Color32::WHITE } else { text_color },
        );
    };
    draw_trigger(lt_pos, gamepad.lt, "LT");
    draw_trigger(rt_pos, gamepad.rt, "RT");

    let bumper_size = Vec2::new(40.0, 14.0);
    let lb_pos = lt_pos + Vec2::new(trigger_size.x / 2.0 + bumper_size.x / 2.0 + 2.0, 0.0);
    let rb_pos = rt_pos - Vec2::new(trigger_size.x / 2.0 + bumper_size.x / 2.0 + 2.0, 0.0);

    let draw_bumper = |center: Pos2, active: bool, label: &str| {
        let r = Rect::from_center_size(center, bumper_size);
        let fill = if active {
            Color32::from_rgb(50, 200, 200)
        } else {
            btn_base_color
        };
        painter.rect_filled(r, Rounding::same(2.0), fill);
        painter.rect_stroke(r, Rounding::same(2.0), Stroke::new(1.0, outline_color));
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(10.0),
            if active { Color32::BLACK } else { text_color },
        );
    };
    draw_bumper(lb_pos, gamepad.lb, "LB");
    draw_bumper(rb_pos, gamepad.rb, "RB");

    // 2. Face View
    let face_rect = Rect::from_min_size(
        top_left + Vec2::new(0.0, height_back + spacing),
        Vec2::new(width, height_front),
    );
    painter.text(
        face_rect.min + Vec2::new(5.0, 0.0),
        egui::Align2::LEFT_TOP,
        "FACE VIEW",
        egui::FontId::proportional(10.0),
        text_color,
    );
    let fc = face_rect.center();

    let body_points = vec![
        fc + Vec2::new(-70.0, -40.0),
        fc + Vec2::new(70.0, -40.0),
        fc + Vec2::new(110.0, 20.0),
        fc + Vec2::new(70.0, 60.0),
        fc + Vec2::new(-70.0, 60.0),
        fc + Vec2::new(-110.0, 20.0),
    ];
    painter.add(Shape::convex_polygon(
        body_points.clone(),
        body_color,
        Stroke::new(1.5, outline_color),
    ));

    let draw_stick = |c: Pos2, x: f32, y: f32, lbl: &str| {
        painter.circle_filled(c, 22.0, btn_base_color);
        painter.circle_stroke(c, 22.0, Stroke::new(1.0, outline_color));
        let head = c + Vec2::new(x, -y) * 12.0;
        let act = x.abs() > 0.1 || y.abs() > 0.1;
        let col = if act {
            Color32::from_rgb(0, 255, 255)
        } else {
            Color32::from_rgb(60, 60, 65)
        };
        painter.circle_filled(head, 14.0, col);
        painter.circle_stroke(head, 14.0, Stroke::new(1.0, outline_color));
        painter.text(
            c + Vec2::new(0.0, 35.0),
            egui::Align2::CENTER_TOP,
            lbl,
            egui::FontId::proportional(12.0),
            text_color,
        );
    };
    draw_stick(fc + Vec2::new(-65.0, -10.0), gamepad.lx, gamepad.ly, "LS");
    draw_stick(fc + Vec2::new(35.0, 30.0), gamepad.rx, gamepad.ry, "RS");

    let dpad_c = fc + Vec2::new(-35.0, 30.0);
    let d_sz = 10.0;
    let draw_dpad_arm = |offset: Vec2, active: bool| {
        let r = Rect::from_center_size(dpad_c + offset, Vec2::splat(d_sz));
        let c = if active {
            Color32::from_rgb(255, 165, 0)
        } else {
            btn_base_color
        };
        painter.rect_filled(r, Rounding::same(2.0), c);
        painter.rect_stroke(r, Rounding::same(2.0), Stroke::new(1.0, outline_color));
    };
    draw_dpad_arm(Vec2::new(0.0, 0.0), false);
    draw_dpad_arm(Vec2::new(0.0, -d_sz), gamepad.dpad_up);
    draw_dpad_arm(Vec2::new(0.0, d_sz), gamepad.dpad_down);
    draw_dpad_arm(Vec2::new(-d_sz, 0.0), gamepad.dpad_left);
    draw_dpad_arm(Vec2::new(d_sz, 0.0), gamepad.dpad_right);

    let btn_c = fc + Vec2::new(65.0, -30.0);
    let b_rad = 11.0;
    let b_gap = 20.0;
    let draw_face_btn = |offset: Vec2, active: bool, label: &str, color: Color32| {
        let pos = btn_c + offset;
        let fill = if active { color } else { btn_base_color };
        painter.circle_filled(pos, b_rad, fill);
        painter.circle_stroke(pos, b_rad, Stroke::new(1.0, outline_color));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(14.0),
            if active { Color32::BLACK } else { color },
        );
    };
    draw_face_btn(Vec2::new(0.0, b_gap), gamepad.a, "A", Color32::GREEN);
    draw_face_btn(Vec2::new(b_gap, 0.0), gamepad.b, "B", Color32::RED);
    draw_face_btn(Vec2::new(-b_gap, 0.0), gamepad.x, "X", Color32::BLUE);
    draw_face_btn(Vec2::new(0.0, -b_gap), gamepad.y, "Y", Color32::YELLOW);
}
