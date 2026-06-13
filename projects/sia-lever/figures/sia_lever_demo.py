"""
Manim demo of the SIA-Lever rotation testbed.

Story:
  1. The task: rotate a 2D point by angle Delta.
  2. The trap: a shortcut channel leaks the answer; a cheater copies it.
  3. The model: encoder -> A(Delta) -> decoder, with A the testable group action.
  4. The two levers: W (weights) vs H (harness/verifier).
  5. The lever phenomenon: W-only keeps the cheat; H->W repairs it.

Render:
  manim -qm figures/sia_lever_demo.py SIALeverDemo
"""

from manim import *
import numpy as np


BLUE_C = "#5b8def"
GREEN_C = "#3cb371"
PURPLE_C = "#9b59b6"
ORANGE_C = "#e67e22"
RED_C = "#e74c3c"
GRAY_C = "#7f8c8d"


class SIALeverDemo(Scene):
    def construct(self):
        self.title_card()
        self.task_scene()
        self.shortcut_trap()
        self.model_scene()
        self.lever_scene()
        self.outro()

    # ------------------------------------------------------------------ #
    def title_card(self):
        t = Text("SIA-Lever", font_size=64, weight=BOLD)
        s = Text("which lever fixes a failing agent: weights or harness?",
                 font_size=26, color=GRAY_C)
        s.next_to(t, DOWN, buff=0.4)
        self.play(Write(t))
        self.play(FadeIn(s, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(t), FadeOut(s))

    # ------------------------------------------------------------------ #
    def task_scene(self):
        head = Text("The task: rotate a point by angle Delta",
                    font_size=34, weight=BOLD).to_edge(UP)
        self.play(FadeIn(head))

        plane = NumberPlane(
            x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1],
            x_length=5, y_length=5,
            background_line_style={"stroke_opacity": 0.25},
        ).shift(DOWN * 0.3)
        self.play(Create(plane))

        r = 1.8
        theta = 30 * DEGREES
        delta = 80 * DEGREES
        origin = plane.c2p(0, 0)

        x_pt = plane.c2p(r * np.cos(theta), r * np.sin(theta))
        y_pt = plane.c2p(r * np.cos(theta + delta), r * np.sin(theta + delta))

        x_vec = Arrow(origin, x_pt, buff=0, color=BLUE_C, stroke_width=6)
        x_lbl = MathTex("x", color=BLUE_C).next_to(x_pt, UR, buff=0.1)

        self.play(GrowArrow(x_vec), FadeIn(x_lbl))
        self.wait(0.5)

        arc = Arc(radius=0.6, start_angle=theta, angle=delta,
                  arc_center=origin, color=YELLOW)
        d_lbl = MathTex(r"\Delta", color=YELLOW).move_to(
            origin + 1.0 * np.array([np.cos(theta + delta / 2),
                                     np.sin(theta + delta / 2), 0]))

        y_vec = Arrow(origin, y_pt, buff=0, color=GREEN_C, stroke_width=6)
        y_lbl = MathTex(r"y=\mathrm{rotate}(x,\Delta)",
                        color=GREEN_C).next_to(y_pt, UL, buff=0.1)

        self.play(Create(arc), FadeIn(d_lbl))
        self.play(Rotate(x_vec.copy(), angle=delta, about_point=origin,
                         rate_func=smooth, run_time=1.2),
                  GrowArrow(y_vec))
        self.play(FadeIn(y_lbl))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in
                    [head, plane, x_vec, x_lbl, arc, d_lbl, y_vec, y_lbl]])

    # ------------------------------------------------------------------ #
    def shortcut_trap(self):
        head = Text("The trap: a shortcut channel leaks the answer",
                    font_size=34, weight=BOLD).to_edge(UP)
        self.play(FadeIn(head))

        inp = self._box("Input (6-vector)\n[x, sin D, cos D, shortcut]",
                        ORANGE_C, width=4.2).shift(LEFT * 4)
        leak = Text("shortcut ~ y  (the target!)",
                    font_size=24, color=RED_C).next_to(inp, DOWN, buff=0.6)

        self.play(FadeIn(inp))
        self.play(FadeIn(leak))
        self.wait(0.8)

        honest = self._box("Honest learner\nlearns rotation", GREEN_C,
                           width=3.4).shift(RIGHT * 2 + UP * 1.3)
        cheat = self._box("Cheater\ncopies shortcut", RED_C,
                          width=3.4).shift(RIGHT * 2 + DOWN * 1.3)

        a1 = Arrow(inp.get_right(), honest.get_left(), color=GREEN_C, buff=0.2)
        a2 = Arrow(inp.get_right(), cheat.get_left(), color=RED_C, buff=0.2)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(honest), FadeIn(cheat))
        self.wait(0.5)

        note = Text("Both score ~0 prediction error. MSE cannot tell them apart.",
                    font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(2)

        self.play(*[FadeOut(m) for m in
                    [head, inp, leak, honest, cheat, a1, a2, note]])

    # ------------------------------------------------------------------ #
    def model_scene(self):
        head = Text("The model: encoder -> A(Delta) -> decoder",
                    font_size=34, weight=BOLD).to_edge(UP)
        self.play(FadeIn(head))

        inp = self._box("Input\n6-vec", ORANGE_C, width=1.8).shift(LEFT * 5.2)
        enc = self._box("Encoder\nz", BLUE_C, width=1.8).shift(LEFT * 2.6)
        act = self._box("ActionNet\nA(D)", GREEN_C, width=2.0).shift(UP * 2.2)
        ap = self._box("z' = A z", PURPLE_C, width=1.8).shift(RIGHT * 0.4)
        dec = self._box("Decoder", PURPLE_C, width=1.8).shift(RIGHT * 3.0)
        out = self._box("y_hat", ORANGE_C, width=1.6).shift(RIGHT * 5.4)

        for m in [inp, enc, ap, dec, out]:
            self.play(FadeIn(m), run_time=0.3)
        self.play(FadeIn(act), run_time=0.3)

        arrows = [
            Arrow(inp.get_right(), enc.get_left(), buff=0.1, color=WHITE),
            Arrow(enc.get_right(), ap.get_left(), buff=0.1, color=WHITE),
            Arrow(act.get_bottom(), ap.get_top(), buff=0.1, color=GREEN_C),
            Arrow(ap.get_right(), dec.get_left(), buff=0.1, color=WHITE),
            Arrow(dec.get_right(), out.get_left(), buff=0.1, color=WHITE),
        ]
        # angle-only path into ActionNet
        ang = Arrow(inp.get_top(), act.get_left(), buff=0.1, color=GREEN_C)
        self.play(*[GrowArrow(a) for a in arrows], GrowArrow(ang))
        self.wait(0.5)

        note = Text("A(Delta) sees only the angle. If the model cheats via the\n"
                    "shortcut, A stays garbage -> group axioms expose it.",
                    font_size=22, color=YELLOW).to_edge(DOWN)
        self.play(FadeIn(note))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in
                    [head, inp, enc, act, ap, dec, out, note, ang] + arrows])

    # ------------------------------------------------------------------ #
    def lever_scene(self):
        head = Text("Two levers", font_size=34, weight=BOLD).to_edge(UP)
        self.play(FadeIn(head))

        w = self._box("Weights (W)\nencoder + action_net\n+ decoder params",
                      GREEN_C, width=4.2).shift(LEFT * 3.2)
        h = self._box("Harness (H)\nobjective + verifier",
                      GRAY_C, width=4.2).shift(RIGHT * 3.2)
        self.play(FadeIn(w), FadeIn(h))

        weak = self._box("weak: prediction MSE\n(fooled by shortcut)", RED_C,
                         width=4.0, fs=22).next_to(h, DOWN, buff=0.5)
        strong = self._box("structural: neg_control,\nshortcut_sens, comp/id/inv",
                           GREEN_C, width=4.0, fs=22).next_to(weak, DOWN, buff=0.3)
        self.play(FadeIn(weak))
        self.wait(0.4)
        self.play(FadeIn(strong))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in [head, w, h, weak, strong]])

        # phenomenon bars
        head2 = Text("Pull the wrong lever -> cheat stays. H then W -> repaired.",
                     font_size=30, weight=BOLD).to_edge(UP)
        self.play(FadeIn(head2))

        stages = [
            ("pred-only", 0.12, RED_C, "shortcut win"),
            ("W-only", 0.145, RED_C, "shortcut win"),
            ("H->W", 1.067, GREEN_C, "clean win"),
        ]
        bars = VGroup()
        max_h = 3.0
        max_v = 1.1
        for i, (name, val, col, verdict) in enumerate(stages):
            bh = max_h * (val / max_v)
            bar = Rectangle(width=1.4, height=bh, fill_color=col,
                            fill_opacity=0.85, stroke_width=0)
            bar.move_to(np.array([-3.5 + i * 3.5, -2.0 + bh / 2, 0]))
            nm = Text(name, font_size=22).next_to(bar, DOWN, buff=0.2)
            vd = Text(verdict, font_size=18, color=col).next_to(bar, UP, buff=0.15)
            bars.add(VGroup(bar, nm, vd))

        ylab = Text("neg_control_mse  (high = honest)", font_size=20,
                    color=YELLOW).to_edge(LEFT).shift(UP * 1.5).rotate(0)
        self.play(FadeIn(ylab))
        for g in bars:
            self.play(GrowFromEdge(g[0], DOWN), FadeIn(g[1]), FadeIn(g[2]),
                      run_time=0.7)
        self.wait(2)

        self.play(*[FadeOut(m) for m in [head2, bars, ylab]])

    # ------------------------------------------------------------------ #
    def outro(self):
        t = Text("Wrong lever entrenches failure.\nFix the harness, then the weights.",
                 font_size=34, weight=BOLD)
        sub = Text("W-only vs H->W:  shortcut_sensitivity  t=44.6, p=1.6e-16",
                   font_size=24, color=GRAY_C).next_to(t, DOWN, buff=0.5)
        self.play(Write(t))
        self.play(FadeIn(sub))
        self.wait(2.5)
        self.play(FadeOut(t), FadeOut(sub))

    # ------------------------------------------------------------------ #
    def _box(self, label, color, width=2.0, height=1.1, fs=24):
        rect = RoundedRectangle(corner_radius=0.12, width=width, height=height,
                                stroke_color=color, stroke_width=3,
                                fill_color=color, fill_opacity=0.15)
        txt = Text(label, font_size=fs, line_spacing=0.8).move_to(rect)
        if txt.width > width - 0.3:
            txt.scale((width - 0.3) / txt.width)
        return VGroup(rect, txt)
