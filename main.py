# =============================================================================
# Segmentação e Classificação de Imagens Mamográficas
# Disciplina: Processamento e Análise de Imagens - PUC Minas
# Prof. Alexei Machado
#
# Grupo: [PREENCHER COM NOME, MATRÍCULA, CURSO E CAMPUS DOS INTEGRANTES]
# =============================================================================

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import time
import random
import math
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np


# ─────────────────────────────────────────────
#  Aparência global
# ─────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Paleta de cores personalizada
COLORS_DARK = {
    "bg_deep":      "#0a0e1a",
    "bg_panel":     "#111827",
    "bg_card":      "#1a2235",
    "bg_hover":     "#1f2d45",
    "accent_blue":  "#3b82f6",
    "accent_cyan":  "#06b6d4",
    "accent_green": "#10b981",
    "accent_red":   "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_purple":"#8b5cf6",
    "text_primary": "#f1f5f9",
    "text_secondary":"#94a3b8",
    "text_muted":   "#475569",
    "border":       "#1e3a5f",
    "border_light": "#2d4a7a",
}

COLORS_LIGHT = {
    "bg_deep":      "#f8fafc",
    "bg_panel":     "#f1f5f9",
    "bg_card":      "#e2e8f0",
    "bg_hover":     "#cbd5e1",
    "accent_blue":  "#2563eb",
    "accent_cyan":  "#0891b2",
    "accent_green": "#059669",
    "accent_red":   "#dc2626",
    "accent_amber": "#d97706",
    "accent_purple":"#7c3aed",
    "text_primary": "#1e293b",
    "text_secondary":"#475569",
    "text_muted":   "#94a3b8",
    "border":       "#cbd5e1",
    "border_light": "#e2e8f0",
}

def get_colors():
    """Retorna a paleta de cores baseado no modo atual."""
    if ctk.get_appearance_mode() == "Dark":
        return COLORS_DARK
    else:
        return COLORS_LIGHT

COLORS = get_colors()

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEADER = ("Courier New", 13, "bold")
FONT_BODY   = ("Courier New", 11)
FONT_SMALL  = ("Courier New", 9)
FONT_MONO   = ("Courier New", 10)
FONT_METRIC = ("Courier New", 20, "bold")


# =============================================================================
#  Widgets auxiliares
# =============================================================================

class SectionLabel(ctk.CTkLabel):
    """Rótulo de seção com linha decorativa."""
    def __init__(self, master, text, **kw):
        super().__init__(
            master,
            text=f"▸ {text}",
            font=FONT_HEADER,
            text_color=COLORS["accent_cyan"],
            anchor="w",
            **kw,
        )


class MetricCard(ctk.CTkFrame):
    """Cartão para exibir uma única métrica."""
    def __init__(self, master, label: str, value: str = "—", color: str = None, **kw):
        super().__init__(
            master,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
            **kw,
        )
        self._color = color or COLORS["accent_cyan"]
        self._lbl_var = ctk.StringVar(value=value)

        ctk.CTkLabel(self, text=label, font=FONT_SMALL,
                     text_color=COLORS["text_secondary"]).pack(pady=(10, 0))
        ctk.CTkLabel(self, textvariable=self._lbl_var,
                     font=FONT_METRIC, text_color=self._color).pack(pady=(0, 10))

    def set_value(self, v: str):
        self._lbl_var.set(v)


class StatusBar(ctk.CTkFrame):
    """Barra de status inferior."""
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=COLORS["bg_panel"],
                         corner_radius=0, height=28, **kw)
        self._var = ctk.StringVar(value="Pronto.")
        ctk.CTkLabel(self, textvariable=self._var,
                     font=FONT_SMALL, text_color=COLORS["text_secondary"],
                     anchor="w").pack(side="left", padx=12)
        self._dot = ctk.CTkLabel(self, text="●", font=FONT_SMALL,
                                 text_color=COLORS["accent_green"])
        self._dot.pack(side="right", padx=12)

    def set(self, msg: str, level: str = "ok"):
        self._var.set(msg)
        cores = {"ok": COLORS["accent_green"],
                 "warn": COLORS["accent_amber"],
                 "err": COLORS["accent_red"],
                 "info": COLORS["accent_cyan"]}
        self._dot.configure(text_color=cores.get(level, COLORS["accent_green"]))


# =============================================================================
#  Aba 1 – Visualizador de Imagem
# =============================================================================

class TabVisualizador(ctk.CTkFrame):
    def __init__(self, master, status_bar: StatusBar, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.status = status_bar
        self._img_original = None
        self._img_segmented = None
        self._zoom = 1.0
        self._show_mask = False
        self._build()

    # ── layout ──────────────────────────────────────────────────────────────
    def _build(self):
        # Painel esquerdo de controles
        left = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                            corner_radius=12, width=240)
        left.pack(side="left", fill="y", padx=(0, 8), pady=0)
        left.pack_propagate(False)

        SectionLabel(left, "CARREGAR IMAGEM").pack(anchor="w", padx=16, pady=(16, 6))

        ctk.CTkButton(left, text="📂  Abrir PNG / TIFF",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_blue"],
                      hover_color=COLORS["bg_hover"],
                      command=self._open_image).pack(padx=16, pady=4, fill="x")

        SectionLabel(left, "ZOOM").pack(anchor="w", padx=16, pady=(18, 6))
        self._zoom_label = ctk.CTkLabel(left, text="100 %",
                                        font=FONT_BODY,
                                        text_color=COLORS["text_primary"])
        self._zoom_label.pack()
        self._zoom_slider = ctk.CTkSlider(left, from_=0.2, to=4.0,
                                          number_of_steps=38,
                                          command=self._on_zoom,
                                          button_color=COLORS["accent_cyan"],
                                          progress_color=COLORS["accent_blue"])
        self._zoom_slider.set(1.0)
        self._zoom_slider.pack(padx=16, fill="x", pady=4)

        ctk.CTkButton(left, text="1:1  Reset zoom", font=FONT_SMALL,
                      fg_color=COLORS["bg_card"],
                      hover_color=COLORS["bg_hover"],
                      command=self._reset_zoom).pack(padx=16, pady=2, fill="x")

        SectionLabel(left, "SEGMENTAÇÃO").pack(anchor="w", padx=16, pady=(18, 6))
        ctk.CTkButton(left, text="⚙  Segmentar Mama",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_purple"],
                      hover_color=COLORS["bg_hover"],
                      command=self._segment).pack(padx=16, pady=4, fill="x")

        self._toggle_mask = ctk.CTkButton(left,
                                          text="👁  Mostrar Máscara",
                                          font=FONT_SMALL,
                                          fg_color=COLORS["bg_card"],
                                          hover_color=COLORS["bg_hover"],
                                          command=self._toggle_mask_view,
                                          state="disabled")
        self._toggle_mask.pack(padx=16, pady=2, fill="x")

        SectionLabel(left, "INFO").pack(anchor="w", padx=16, pady=(18, 6))
        self._info_box = ctk.CTkTextbox(left, height=140,
                                        font=FONT_MONO,
                                        fg_color=COLORS["bg_card"],
                                        text_color=COLORS["text_secondary"],
                                        state="disabled")
        self._info_box.pack(padx=16, fill="x")

        # Área da imagem (direita)
        right = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        # cabeçalho
        hdr = ctk.CTkFrame(right, fg_color="transparent", height=42)
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        hdr.pack_propagate(False)
        self._img_title = ctk.CTkLabel(hdr, text="Nenhuma imagem carregada",
                                       font=FONT_HEADER,
                                       text_color=COLORS["text_secondary"],
                                       anchor="w")
        self._img_title.pack(side="left")

        # Canvas com scroll
        canvas_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_deep"],
                                    corner_radius=8)
        canvas_frame.pack(fill="both", expand=True, padx=12, pady=12)

        self._canvas = tk.Canvas(canvas_frame,
                                 bg=COLORS["bg_deep"],
                                 highlightthickness=0)
        v_scroll = ctk.CTkScrollbar(canvas_frame, orientation="vertical",
                                    command=self._canvas.yview)
        h_scroll = ctk.CTkScrollbar(canvas_frame, orientation="horizontal",
                                    command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=v_scroll.set,
                               xscrollcommand=h_scroll.set)
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right",  fill="y")
        self._canvas.pack(fill="both", expand=True)

        # grade de fundo pontilhada
        self._canvas.bind("<Configure>", self._draw_bg_grid)

    # ── callbacks ────────────────────────────────────────────────────────────
    def _draw_bg_grid(self, event=None):
        if self._img_original is None:
            self._canvas.delete("grid")
            w = self._canvas.winfo_width()
            h = self._canvas.winfo_height()
            step = 32
            for x in range(0, w, step):
                for y in range(0, h, step):
                    self._canvas.create_oval(x, y, x+1, y+1,
                                             fill=COLORS["text_muted"],
                                             outline="", tags="grid")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagem mamográfica",
            filetypes=[("Imagens", "*.png *.tif *.tiff"),
                       ("PNG", "*.png"),
                       ("TIFF", "*.tif *.tiff"),
                       ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
            self._img_original = img
            self._img_segmented = None
            self._show_mask = False
            self._toggle_mask.configure(state="disabled")
            self._zoom = 1.0
            self._zoom_slider.set(1.0)
            self._zoom_label.configure(text="100 %")
            self._img_title.configure(
                text=os.path.basename(path),
                text_color=COLORS["text_primary"]
            )
            self._update_info(path, img)
            self._render()
            self.status.set(f"Imagem carregada: {os.path.basename(path)}", "ok")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{e}")
            self.status.set("Erro ao carregar imagem.", "err")

    def _update_info(self, path, img):
        mode_map = {
            "L": "Tons de cinza (8-bit)",
            "I;16": "16-bit grayscale",
            "I": "32-bit int",
            "F": "32-bit float",
            "RGB": "RGB colorida",
        }
        mode_str = mode_map.get(img.mode, img.mode)
        size_kb   = os.path.getsize(path) / 1024
        info = (
            f"Arquivo : {os.path.basename(path)}\n"
            f"Tamanho : {img.width} × {img.height} px\n"
            f"Modo    : {mode_str}\n"
            f"Disco   : {size_kb:.1f} KB\n"
            f"Formato : {img.format or 'desconhecido'}\n"
        )
        self._info_box.configure(state="normal")
        self._info_box.delete("1.0", "end")
        self._info_box.insert("1.0", info)
        self._info_box.configure(state="disabled")

    def _on_zoom(self, val):
        self._zoom = float(val)
        self._zoom_label.configure(text=f"{int(self._zoom * 100)} %")
        self._render()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._zoom_slider.set(1.0)
        self._zoom_label.configure(text="100 %")
        self._render()

    def _render(self):
        if self._img_original is None:
            return
        src = self._img_segmented if (self._show_mask and self._img_segmented) \
              else self._img_original

        # Normaliza para 8-bit para exibição
        arr = np.array(src)
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
        pil_8 = Image.fromarray(arr)
        if pil_8.mode not in ("L", "RGB", "RGBA"):
            pil_8 = pil_8.convert("L")

        w = max(1, int(pil_8.width  * self._zoom))
        h = max(1, int(pil_8.height * self._zoom))
        pil_resized = pil_8.resize((w, h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil_resized)
        self._canvas.delete("img")
        self._canvas.create_image(0, 0, anchor="nw",
                                  image=self._tk_img, tags="img")
        self._canvas.configure(scrollregion=(0, 0, w, h))

    def _segment(self):
        if self._img_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        self.status.set("Segmentando…", "info")
        threading.Thread(target=self._run_segmentation, daemon=True).start()

    def _run_segmentation(self):
        """
        Segmentação automática da região da mama.
        Estratégia: limiarização de Otsu + morfologia para remover fundo e anotações.
        """
        time.sleep(0.3)   # simula processamento para feedback visual
        try:
            arr = np.array(self._img_original)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
            if len(arr.shape) == 3:
                arr = arr[:, :, 0]

            # --- Otsu threshold simples ---
            hist, bins = np.histogram(arr.flatten(), 256, [0, 256])
            total = arr.size
            sum_all = float(np.dot(np.arange(256), hist))
            w0 = 0.0; sum0 = 0.0; best_var = 0.0; thresh = 0
            for t in range(256):
                w0 += hist[t]
                if w0 == 0:
                    continue
                w1 = total - w0
                if w1 == 0:
                    break
                sum0 += t * hist[t]
                m0 = sum0 / w0
                m1 = (sum_all - sum0) / w1
                var = w0 * w1 * (m0 - m1) ** 2
                if var > best_var:
                    best_var = var
                    thresh = t

            mask = (arr > thresh).astype(np.uint8) * 255

            # Erosão simples para remover anotações finas
            kernel_size = 5
            from PIL import ImageFilter
            pil_mask = Image.fromarray(mask)
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(kernel_size))
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(kernel_size * 3))

            mask_arr = np.array(pil_mask)
            result = np.where(mask_arr > 0, arr, 0).astype(np.uint8)
            self._img_segmented = Image.fromarray(result)

            self.after(0, self._on_segment_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro", str(e)))
            self.after(0, lambda: self.status.set("Erro na segmentação.", "err"))

    def _on_segment_done(self):
        self._toggle_mask.configure(state="normal")
        self._show_mask = True
        self._toggle_mask.configure(text="👁  Mostrar Original")
        self._render()
        self.status.set("Segmentação concluída.", "ok")

    def _toggle_mask_view(self):
        self._show_mask = not self._show_mask
        self._toggle_mask.configure(
            text="👁  Mostrar Original" if self._show_mask else "👁  Mostrar Máscara"
        )
        self._render()


# =============================================================================
#  Aba 2 – Dataset e Aumento de Dados
# =============================================================================

class TabDataset(ctk.CTkFrame):
    def __init__(self, master, status_bar: StatusBar, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.status = status_bar
        self._train_dir = ""
        self._test_dir  = ""
        self._train_imgs: list[str] = []
        self._test_imgs:  list[str] = []
        self._build()

    def _build(self):
        # ── Linha superior: seleção de diretórios ────────────────────────────
        top = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        top.pack(fill="x", pady=(0, 8))

        SectionLabel(top, "DIRETÓRIO DO DATASET").pack(anchor="w", padx=16, pady=(14, 8))

        dir_row = ctk.CTkFrame(top, fg_color="transparent")
        dir_row.pack(fill="x", padx=16, pady=(0, 14))

        # botão carregar diretório (auto split treino/teste)
        ctk.CTkButton(dir_row, text="📁  Selecionar Diretório",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_blue"],
                      hover_color=COLORS["bg_hover"],
                      command=self._load_dir).pack(side="left", padx=(0, 12))

        self._dir_label = ctk.CTkLabel(dir_row, text="Nenhum diretório selecionado",
                                       font=FONT_SMALL,
                                       text_color=COLORS["text_secondary"],
                                       anchor="w")
        self._dir_label.pack(side="left", fill="x", expand=True)

        # ── Estatísticas por classe ──────────────────────────────────────────
        stats_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        stats_frame.pack(fill="x", pady=(0, 8))
        SectionLabel(stats_frame, "DISTRIBUIÇÃO DAS CLASSES (BIRADS)").pack(
            anchor="w", padx=16, pady=(14, 8))

        cards_row = ctk.CTkFrame(stats_frame, fg_color="transparent")
        cards_row.pack(fill="x", padx=16, pady=(0, 14))

        birads_colors = [COLORS["accent_cyan"], COLORS["accent_blue"],
                         COLORS["accent_purple"], COLORS["accent_amber"]]
        self._class_cards: list[MetricCard] = []
        for i, (lbl, col) in enumerate(zip(
                ["BIRADS I\n(Gordura)", "BIRADS II\n(Fibrogland.)",
                 "BIRADS III\n(Denso-Het.)", "BIRADS IV\n(Ext. Denso)"],
                birads_colors)):
            mc = MetricCard(cards_row, label=lbl, color=col)
            mc.pack(side="left", fill="both", expand=True, padx=4)
            self._class_cards.append(mc)

        split_row = ctk.CTkFrame(stats_frame, fg_color="transparent")
        split_row.pack(fill="x", padx=16, pady=(0, 14))
        self._train_card = MetricCard(split_row, "Treino", color=COLORS["accent_green"])
        self._train_card.pack(side="left", fill="both", expand=True, padx=4)
        self._test_card  = MetricCard(split_row, "Teste (múlt. de 4)",
                                      color=COLORS["accent_amber"])
        self._test_card.pack(side="left", fill="both", expand=True, padx=4)
        self._total_card = MetricCard(split_row, "Total", color=COLORS["accent_cyan"])
        self._total_card.pack(side="left", fill="both", expand=True, padx=4)

        # ── Aumento de dados ─────────────────────────────────────────────────
        aug_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        aug_frame.pack(fill="x", pady=(0, 8))
        SectionLabel(aug_frame, "AUMENTO DE DADOS (DATA AUGMENTATION)").pack(
            anchor="w", padx=16, pady=(14, 6))

        aug_info = ctk.CTkLabel(aug_frame,
            text="Rotações: −20°  −10°  0°  +10°  +20°  →  5× por imagem de treino",
            font=FONT_BODY, text_color=COLORS["text_secondary"])
        aug_info.pack(anchor="w", padx=16, pady=(0, 8))

        aug_btn_row = ctk.CTkFrame(aug_frame, fg_color="transparent")
        aug_btn_row.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(aug_btn_row, text="⟳  Realizar Aumento",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_green"],
                      hover_color=COLORS["bg_hover"],
                      command=self._augment).pack(side="left", padx=(0, 12))

        self._aug_progress = ctk.CTkProgressBar(aug_btn_row,
                                                mode="determinate",
                                                progress_color=COLORS["accent_green"])
        self._aug_progress.set(0)
        self._aug_progress.pack(side="left", fill="x", expand=True)

        self._aug_label = ctk.CTkLabel(aug_btn_row, text="0 / 0",
                                       font=FONT_SMALL,
                                       text_color=COLORS["text_secondary"],
                                       width=60)
        self._aug_label.pack(side="left", padx=8)

        # ── Log ──────────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        log_frame.pack(fill="both", expand=True)
        SectionLabel(log_frame, "LOG").pack(anchor="w", padx=16, pady=(14, 6))
        self._log = ctk.CTkTextbox(log_frame, font=FONT_MONO,
                                   fg_color=COLORS["bg_deep"],
                                   text_color=COLORS["accent_cyan"],
                                   state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ── callbacks ────────────────────────────────────────────────────────────
    def _log_msg(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _load_dir(self):
        d = filedialog.askdirectory(title="Selecionar diretório com imagens")
        if not d:
            return
        self._train_dir = d
        self._dir_label.configure(text=d, text_color=COLORS["text_primary"])
        self._scan_dir(d)

    def _scan_dir(self, d: str):
        """Varre diretório e separa treino/teste conforme especificação."""
        exts = {".png", ".tif", ".tiff"}
        all_imgs = []
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in exts:
                    all_imgs.append(os.path.join(root, f))

        if not all_imgs:
            messagebox.showwarning("Aviso", "Nenhuma imagem PNG/TIFF encontrada.")
            return

        # Split: múltiplo de 4 → teste, demais → treino
        train, test = [], []
        for path in all_imgs:
            stem = os.path.splitext(os.path.basename(path))[0]
            digits = "".join(c for c in stem if c.isdigit())
            num = int(digits) if digits else 0
            (test if num % 4 == 0 else train).append(path)

        self._train_imgs = train
        self._test_imgs  = test

        # Contagem por classe (D=I, E=II, F=III, G=IV)
        cls_map = {"D": 0, "E": 1, "F": 2, "G": 3}
        cls_cnt = [0, 0, 0, 0]
        for p in all_imgs:
            first = os.path.basename(p)[0].upper()
            if first in cls_map:
                cls_cnt[cls_map[first]] += 1

        for i, mc in enumerate(self._class_cards):
            mc.set_value(str(cls_cnt[i]))
        self._train_card.set_value(str(len(train)))
        self._test_card.set_value(str(len(test)))
        self._total_card.set_value(str(len(all_imgs)))

        self._log_msg(f"[OK] Diretório: {d}")
        self._log_msg(f"     Total  : {len(all_imgs)} imagens")
        self._log_msg(f"     Treino : {len(train)} | Teste: {len(test)}")
        for k, v in cls_map.items():
            self._log_msg(f"     BIRADS {['I','II','III','IV'][v]}: {cls_cnt[v]}")
        self.status.set(f"Dataset carregado: {len(all_imgs)} imagens.", "ok")

    def _augment(self):
        if not self._train_imgs:
            messagebox.showwarning("Aviso", "Carregue um dataset primeiro.")
            return
        threading.Thread(target=self._run_augment, daemon=True).start()

    def _run_augment(self):
        angles = [-20, -10, 0, 10, 20]
        total  = len(self._train_imgs) * len(angles)
        done   = 0
        self.after(0, lambda: self.status.set("Aumento de dados em progresso…", "info"))

        for img_path in self._train_imgs:
            try:
                img = Image.open(img_path)
            except Exception:
                continue
            for ang in angles:
                rotated = img.rotate(ang, expand=False, fillcolor=0)
                # Em um projeto real salvaria o arquivo; aqui apenas simula
                done += 1
                frac = done / total
                self.after(0, lambda f=frac, d=done, t=total:
                           (self._aug_progress.set(f),
                            self._aug_label.configure(text=f"{d} / {t}")))
                time.sleep(0.005)

        self.after(0, lambda: (
            self._log_msg(f"[OK] Aumento concluído: {total} imagens geradas."),
            self.status.set("Aumento de dados concluído.", "ok")
        ))


# =============================================================================
#  Aba 3 – Classificação
# =============================================================================

class TabClassificacao(ctk.CTkFrame):
    def __init__(self, master, status_bar: StatusBar, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.status = status_bar
        self._mode = ctk.StringVar(value="binario")
        self._build()

    def _build(self):
        # ── Configuração ─────────────────────────────────────────────────────
        cfg = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        cfg.pack(fill="x", pady=(0, 8))
        SectionLabel(cfg, "CONFIGURAÇÃO DO CLASSIFICADOR").pack(
            anchor="w", padx=16, pady=(14, 8))

        row = ctk.CTkFrame(cfg, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(row, text="Modo:", font=FONT_BODY,
                     text_color=COLORS["text_secondary"]).pack(side="left")
        ctk.CTkRadioButton(row, text="Binário (I+II vs III+IV)",
                           variable=self._mode, value="binario",
                           font=FONT_BODY,
                           text_color=COLORS["text_primary"],
                           fg_color=COLORS["accent_blue"]).pack(side="left", padx=20)
        ctk.CTkRadioButton(row, text="4 Classes (I vs II vs III vs IV)",
                           variable=self._mode, value="quadriclasse",
                           font=FONT_BODY,
                           text_color=COLORS["text_primary"],
                           fg_color=COLORS["accent_purple"]).pack(side="left")

        row2 = ctk.CTkFrame(cfg, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(row2, text="▶  Treinar Modelo",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_green"],
                      hover_color=COLORS["bg_hover"],
                      command=self._treinar).pack(side="left", padx=(0, 8))

        ctk.CTkButton(row2, text="⚡  Classificar Teste",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_blue"],
                      hover_color=COLORS["bg_hover"],
                      command=self._classificar).pack(side="left", padx=(0, 8))

        ctk.CTkButton(row2, text="💾  Salvar Modelo",
                      font=FONT_SMALL,
                      fg_color=COLORS["bg_card"],
                      hover_color=COLORS["bg_hover"],
                      command=self._salvar_modelo).pack(side="left")

        # Barra de progresso do treino
        prog_row = ctk.CTkFrame(cfg, fg_color="transparent")
        prog_row.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(prog_row, text="Épocas:", font=FONT_SMALL,
                     text_color=COLORS["text_secondary"]).pack(side="left")
        self._train_prog = ctk.CTkProgressBar(prog_row, mode="determinate",
                                              progress_color=COLORS["accent_green"])
        self._train_prog.set(0)
        self._train_prog.pack(side="left", fill="x", expand=True, padx=8)
        self._epoch_label = ctk.CTkLabel(prog_row, text="—", font=FONT_SMALL,
                                         text_color=COLORS["text_secondary"], width=60)
        self._epoch_label.pack(side="left")
        self._time_label  = ctk.CTkLabel(prog_row, text="", font=FONT_SMALL,
                                         text_color=COLORS["accent_amber"])
        self._time_label.pack(side="left", padx=8)

        # ── Métricas binárias ─────────────────────────────────────────────────
        m_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        m_frame.pack(fill="x", pady=(0, 8))
        SectionLabel(m_frame, "MÉTRICAS – CLASSIFICAÇÃO BINÁRIA").pack(
            anchor="w", padx=16, pady=(14, 8))

        mc_row = ctk.CTkFrame(m_frame, fg_color="transparent")
        mc_row.pack(fill="x", padx=16, pady=(0, 14))
        metric_names = ["Sensib.", "Especif.", "Precisão", "Acurácia", "F1-Score"]
        metric_cols  = [COLORS["accent_cyan"], COLORS["accent_blue"],
                        COLORS["accent_purple"], COLORS["accent_green"],
                        COLORS["accent_amber"]]
        self._bin_metrics: list[MetricCard] = []
        for name, col in zip(metric_names, metric_cols):
            mc = MetricCard(mc_row, label=name, color=col)
            mc.pack(side="left", fill="both", expand=True, padx=4)
            self._bin_metrics.append(mc)

        # ── Matriz de confusão ────────────────────────────────────────────────
        mat_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        mat_frame.pack(fill="both", expand=True, pady=(0, 0))
        SectionLabel(mat_frame, "MATRIZ DE CONFUSÃO – 4 CLASSES").pack(
            anchor="w", padx=16, pady=(14, 8))

        inner = ctk.CTkFrame(mat_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        # Grade 4×4
        labels = ["I", "II", "III", "IV"]
        self._cm_cells: list[list[ctk.CTkLabel]] = []

        header_row = ctk.CTkFrame(inner, fg_color="transparent")
        header_row.pack(fill="x")
        ctk.CTkLabel(header_row, text="Pred →\nReal ↓", font=FONT_SMALL,
                     text_color=COLORS["text_muted"], width=80).pack(side="left")
        for lbl in labels:
            ctk.CTkLabel(header_row, text=f"BIRADS {lbl}",
                         font=FONT_SMALL, text_color=COLORS["accent_cyan"],
                         width=90).pack(side="left", padx=2)

        for i, row_lbl in enumerate(labels):
            r = ctk.CTkFrame(inner, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=f"BIRADS {row_lbl}", font=FONT_SMALL,
                         text_color=COLORS["accent_cyan"], width=80).pack(side="left")
            row_cells = []
            for j in range(4):
                bg = COLORS["bg_card"] if i != j else COLORS["border"]
                cell = ctk.CTkLabel(r, text="—", font=FONT_BODY,
                                    fg_color=bg, corner_radius=6,
                                    text_color=COLORS["text_primary"],
                                    width=86, height=34)
                cell.pack(side="left", padx=2)
                row_cells.append(cell)
            self._cm_cells.append(row_cells)

        # Métricas 4 classes
        m4_row = ctk.CTkFrame(mat_frame, fg_color="transparent")
        m4_row.pack(fill="x", padx=16, pady=(0, 14))
        self._sens_media  = MetricCard(m4_row, "Sensib. Média",
                                       color=COLORS["accent_cyan"])
        self._sens_media.pack(side="left", fill="both", expand=True, padx=4)
        self._espec_media = MetricCard(m4_row, "Especif. Média",
                                       color=COLORS["accent_blue"])
        self._espec_media.pack(side="left", fill="both", expand=True, padx=4)
        self._time_exec   = MetricCard(m4_row, "Tempo de Execução",
                                       color=COLORS["accent_amber"])
        self._time_exec.pack(side="left", fill="both", expand=True, padx=4)

    # ── callbacks ────────────────────────────────────────────────────────────
    def _treinar(self):
        threading.Thread(target=self._run_treino, daemon=True).start()

    def _run_treino(self):
        """Simula progresso de treinamento (preencher com lógica real)."""
        n_epochs = 20
        self.after(0, lambda: self.status.set("Treinando modelo…", "info"))
        t0 = time.time()
        for ep in range(1, n_epochs + 1):
            time.sleep(0.12)   # ← substituir por epoch real
            frac = ep / n_epochs
            elapsed = time.time() - t0
            self.after(0, lambda f=frac, e=ep, t=elapsed: (
                self._train_prog.set(f),
                self._epoch_label.configure(text=f"{e}/{n_epochs}"),
                self._time_label.configure(text=f"{t:.1f}s"),
            ))
        self.after(0, lambda: self.status.set("Treino concluído.", "ok"))

    def _classificar(self):
        """Simula classificação e preenche métricas (preencher com lógica real)."""
        t0 = time.time()
        # Valores de exemplo — substituir pelo resultado real do modelo
        sens  = random.uniform(0.72, 0.92)
        espec = random.uniform(0.74, 0.94)
        prec  = random.uniform(0.70, 0.90)
        acc   = random.uniform(0.75, 0.93)
        f1    = 2 * prec * sens / (prec + sens)
        elapsed = time.time() - t0 + random.uniform(0.5, 3)

        vals = [sens, espec, prec, acc, f1]
        for mc, v in zip(self._bin_metrics, vals):
            mc.set_value(f"{v:.3f}")

        # Matriz de confusão aleatória (simulação)
        cm = np.random.randint(0, 50, (4, 4))
        np.fill_diagonal(cm, np.random.randint(50, 120, 4))
        for i in range(4):
            for j in range(4):
                self._cm_cells[i][j].configure(text=str(cm[i, j]))

        # Métricas médias simuladas
        self._sens_media.set_value(f"{sens:.3f}")
        self._espec_media.set_value(f"{espec:.3f}")
        self._time_exec.set_value(f"{elapsed:.2f}s")
        self.status.set("Classificação concluída.", "ok")

    def _salvar_modelo(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch", "*.pth"), ("H5", "*.h5"), ("Todos", "*.*")]
        )
        if path:
            messagebox.showinfo("Salvar Modelo",
                                f"Modelo seria salvo em:\n{path}\n"
                                "(implemente torch.save() ou model.save() aqui)")
            self.status.set(f"Modelo salvo: {os.path.basename(path)}", "ok")


# =============================================================================
#  Aba 4 – Grad-CAM
# =============================================================================

class TabGradCAM(ctk.CTkFrame):
    def __init__(self, master, status_bar: StatusBar, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.status = status_bar
        self._img_path = ""
        self._build()

    def _build(self):
        # Painel esquerdo
        left = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                            corner_radius=12, width=260)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        SectionLabel(left, "IMAGEM").pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkButton(left, text="📂  Abrir Imagem",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_blue"],
                      hover_color=COLORS["bg_hover"],
                      command=self._open).pack(padx=16, pady=4, fill="x")

        self._img_name = ctk.CTkLabel(left, text="—", font=FONT_SMALL,
                                      text_color=COLORS["text_secondary"],
                                      wraplength=220, anchor="w")
        self._img_name.pack(padx=16, pady=4, anchor="w")

        SectionLabel(left, "RESULTADO").pack(anchor="w", padx=16, pady=(18, 6))
        self._result_card = MetricCard(left, "BIRADS Predito",
                                       color=COLORS["accent_amber"])
        self._result_card.pack(padx=16, fill="x")

        self._conf_card = MetricCard(left, "Confiança",
                                     color=COLORS["accent_green"])
        self._conf_card.pack(padx=16, fill="x", pady=8)

        ctk.CTkButton(left, text="🔥  Gerar Grad-CAM",
                      font=FONT_BODY,
                      fg_color=COLORS["accent_purple"],
                      hover_color=COLORS["bg_hover"],
                      command=self._run_gradcam).pack(padx=16, pady=4, fill="x")

        SectionLabel(left, "LEGENDA").pack(anchor="w", padx=16, pady=(18, 4))
        legend_info = (
            "Azul  → baixa ativação\n"
            "Verde → ativação média\n"
            "Amarelo → alta ativação\n"
            "Vermelho → máx. ativação"
        )
        ctk.CTkLabel(left, text=legend_info, font=FONT_SMALL,
                     text_color=COLORS["text_secondary"],
                     justify="left", anchor="w").pack(padx=16, pady=4, anchor="w")

        # Painel direito: dois canvas lado a lado
        right = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        hdr = ctk.CTkFrame(right, fg_color="transparent", height=42)
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Original", font=FONT_HEADER,
                     text_color=COLORS["text_secondary"],
                     anchor="w").pack(side="left", expand=True)
        ctk.CTkLabel(hdr, text="Grad-CAM Overlay", font=FONT_HEADER,
                     text_color=COLORS["accent_amber"],
                     anchor="w").pack(side="left", expand=True)

        canvases_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_deep"],
                                      corner_radius=8)
        canvases_frame.pack(fill="both", expand=True, padx=12, pady=12)

        self._canvas_orig = tk.Canvas(canvases_frame, bg=COLORS["bg_deep"],
                                      highlightthickness=0)
        self._canvas_cam  = tk.Canvas(canvases_frame, bg=COLORS["bg_deep"],
                                      highlightthickness=0)
        self._canvas_orig.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self._canvas_cam.pack(side="left",  fill="both", expand=True, padx=(4, 0))

    # ── callbacks ────────────────────────────────────────────────────────────
    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")])
        if path:
            self._img_path = path
            self._img_name.configure(text=os.path.basename(path),
                                     text_color=COLORS["text_primary"])
            self._show_original(path)
            self.status.set(f"Imagem: {os.path.basename(path)}", "ok")

    def _show_original(self, path: str):
        try:
            img = Image.open(path)
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
            pil8 = Image.fromarray(arr).convert("L")
            self._orig_pil = pil8
            self._fit_canvas(self._canvas_orig, pil8, "_tk_orig")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _fit_canvas(self, canvas: tk.Canvas, img: Image.Image, attr: str):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(),  200)
        ch = max(canvas.winfo_height(), 200)
        img_r = img.copy()
        img_r.thumbnail((cw, ch), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_r)
        setattr(self, attr, tk_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)

    def _run_gradcam(self):
        if not self._img_path:
            messagebox.showwarning("Aviso", "Selecione uma imagem primeiro.")
            return
        threading.Thread(target=self._generate_gradcam, daemon=True).start()

    def _generate_gradcam(self):
        """Gera um heatmap Grad-CAM sintético (substituir pela implementação real)."""
        self.after(0, lambda: self.status.set("Gerando Grad-CAM…", "info"))
        time.sleep(0.5)
        try:
            orig = self._orig_pil.copy()
            w, h = orig.size

            # Heatmap sintético gaussiano (substituir por gradientes reais da rede)
            heat = np.zeros((h, w), dtype=np.float32)
            cx, cy = w * random.uniform(0.3, 0.7), h * random.uniform(0.3, 0.7)
            for y in range(h):
                for x in range(0, w, max(1, w // 100)):
                    v = math.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(w, h) * 0.2)**2))
                    heat[y, x] = v
            # Interpolar
            from PIL import Image as PILImage
            heat_pil = PILImage.fromarray((heat * 255).astype(np.uint8))
            heat_pil = heat_pil.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 20))
            heat_arr = np.array(heat_pil) / 255.0

            # Colormap jet manual
            def jet(v):
                r = int(np.clip(1.5 - abs(4 * v - 3), 0, 1) * 255)
                g = int(np.clip(1.5 - abs(4 * v - 2), 0, 1) * 255)
                b = int(np.clip(1.5 - abs(4 * v - 1), 0, 1) * 255)
                return r, g, b

            h_arr, w_arr = heat_arr.shape
            cam_rgb = np.zeros((h_arr, w_arr, 3), dtype=np.uint8)
            for y in range(h_arr):
                for x in range(0, w_arr, max(1, w_arr // 200)):
                    cam_rgb[y, x] = jet(heat_arr[y, x])

            cam_pil = Image.fromarray(cam_rgb).resize((w, h), Image.LANCZOS)
            orig_rgb = orig.convert("RGB")
            blended  = Image.blend(orig_rgb, cam_pil, alpha=0.55)

            # Classe e confiança sintéticas (substituir pela saída real da rede)
            birads = random.choice(["I", "II", "III", "IV"])
            conf   = random.uniform(0.70, 0.99)

            self.after(0, lambda b=blended, cl=birads, cf=conf:
                       self._show_gradcam(b, cl, cf))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro Grad-CAM", str(e)))

    def _show_gradcam(self, blended: Image.Image, birads: str, conf: float):
        self._fit_canvas(self._canvas_cam, blended, "_tk_cam")
        self._result_card.set_value(f"BIRADS {birads}")
        self._conf_card.set_value(f"{conf:.1%}")
        self.status.set("Grad-CAM gerado com sucesso.", "ok")


# =============================================================================
#  Janela principal
# =============================================================================

class MamografiaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MamoVision — Segmentação e Classificação Mamográfica")
        self.geometry("1300x820")
        self.minsize(1000, 680)
        self.configure(fg_color=COLORS["bg_deep"])

        self._build()

    def _build(self):
        # ── Cabeçalho ────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="  MamoVision",
            font=FONT_TITLE,
            text_color=COLORS["accent_cyan"],
        ).pack(side="left", padx=8)

        ctk.CTkLabel(
            header,
            text="Segmentação e Classificação de Imagens Mamográficas · PUC Minas",
            font=FONT_SMALL,
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=4)

        # Indicador de aparência
        mode_btn = ctk.CTkButton(
            header, text="☀  Tema Claro", font=FONT_SMALL, width=100,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_hover"],
            command=self._toggle_theme
        )
        mode_btn.pack(side="right", padx=12)
        self._mode_btn = mode_btn

        # ── Status bar ───────────────────────────────────────────────────────
        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        # ── Notebook ─────────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(
            self,
            fg_color=COLORS["bg_deep"],
            segmented_button_fg_color=COLORS["bg_panel"],
            segmented_button_selected_color=COLORS["accent_blue"],
            segmented_button_selected_hover_color=COLORS["accent_cyan"],
            segmented_button_unselected_color=COLORS["bg_panel"],
            segmented_button_unselected_hover_color=COLORS["bg_hover"],
            text_color=COLORS["text_primary"],
            text_color_disabled=COLORS["text_muted"],
        )
        self._tabs.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        tab_names = [
            "📷  Visualizador",
            "📦  Dataset",
            "🧠  Classificação",
            "🔥  Grad-CAM",
        ]
        for name in tab_names:
            self._tabs.add(name)

        # Instanciar abas
        TabVisualizador(self._tabs.tab(tab_names[0]),
                        self._status).pack(fill="both", expand=True)
        TabDataset(self._tabs.tab(tab_names[1]),
                   self._status).pack(fill="both", expand=True)
        TabClassificacao(self._tabs.tab(tab_names[2]),
                         self._status).pack(fill="both", expand=True)
        TabGradCAM(self._tabs.tab(tab_names[3]),
                   self._status).pack(fill="both", expand=True)

    def _toggle_theme(self):
        current = ctk.get_appearance_mode()
        if current == "Dark":
            ctk.set_appearance_mode("Light")
            self._mode_btn.configure(text="🌙  Tema Escuro")
        else:
            ctk.set_appearance_mode("Dark")
            self._mode_btn.configure(text="☀  Tema Claro")
        
        # Atualizar COLORS global
        global COLORS
        COLORS.update(get_colors())
        
        # Recriar a interface com as novas cores
        for widget in self.winfo_children():
            widget.destroy()
        self._build()


# =============================================================================
#  Entry point
# =============================================================================

def main():
    app = MamografiaApp()
    app.mainloop()


if __name__ == "__main__":
    main()