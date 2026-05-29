# =============================================================================
# Segmentação e Classificação de Imagens Mamográficas
# Disciplina: Processamento e Análise de Imagens - PUC Minas
# Prof. Alexei Machado
# Grupo: [NOME, MATRÍCULA, CURSO E CAMPUS DOS INTEGRANTES]
# =============================================================================

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading, time, random, math, os
from PIL import Image, ImageTk, ImageFilter
import numpy as np

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Fontes
FONTE_TITULO  = ("Helvetica", 16, "bold")
FONTE_SECAO   = ("Helvetica", 11, "bold")
FONTE_CORPO   = ("Helvetica", 11)
FONTE_PEQUENA = ("Helvetica", 9)
FONTE_MONO    = ("Courier New", 10)
FONTE_METRICA = ("Helvetica", 18, "bold")


# ── Widgets reutilizáveis ────────────────────────────────────────────────────

def rotulo_secao(pai, texto):
    ctk.CTkLabel(pai, text=texto, font=FONTE_SECAO, anchor="w").pack(
        anchor="w", padx=12, pady=(12, 4))

def botao(pai, texto, comando, **kw):
    return ctk.CTkButton(pai, text=texto, font=FONTE_CORPO,
                         command=comando, **kw)

class CartaoMetrica(ctk.CTkFrame):
    def __init__(self, pai, rotulo, valor="—"):
        super().__init__(pai, corner_radius=8, border_width=1)
        ctk.CTkLabel(self, text=rotulo, font=FONTE_PEQUENA).pack(pady=(8, 0))
        self._var = ctk.StringVar(value=valor)
        ctk.CTkLabel(self, textvariable=self._var, font=FONTE_METRICA).pack(pady=(0, 8))

    def definir(self, v): self._var.set(v)

class BarraStatus(ctk.CTkFrame):
    def __init__(self, pai):
        super().__init__(pai, corner_radius=0, height=26)
        self._var = ctk.StringVar(value="Pronto.")
        ctk.CTkLabel(self, textvariable=self._var, font=FONTE_PEQUENA,
                     anchor="w").pack(side="left", padx=10)

    def definir(self, msg): self._var.set(msg)


# ── Aba Visualizador ─────────────────────────────────────────────────────────

class AbaVisualizador(ctk.CTkFrame):
    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status       = status
        self._img_original  = None
        self._img_segmentada = None
        self._zoom          = 1.0
        self._mostrar_mascara = False
        self._construir()

    def _construir(self):
        # Painel de controles
        painel = ctk.CTkFrame(self, width=220, corner_radius=10)
        painel.pack(side="left", fill="y", padx=(0, 6))
        painel.pack_propagate(False)

        rotulo_secao(painel, "IMAGEM")
        botao(painel, "📂 Abrir PNG/TIFF", self._abrir).pack(padx=12, fill="x")

        rotulo_secao(painel, "ZOOM")
        self._lbl_zoom = ctk.CTkLabel(painel, text="100%", font=FONTE_CORPO)
        self._lbl_zoom.pack()
        self._slider_zoom = ctk.CTkSlider(painel, from_=0.2, to=4.0,
                                          number_of_steps=38, command=self._ao_zoom)
        self._slider_zoom.set(1.0)
        self._slider_zoom.pack(padx=12, fill="x")
        botao(painel, "Reset 1:1", self._reset_zoom,
              fg_color="transparent", border_width=1).pack(padx=12, pady=4, fill="x")

        rotulo_secao(painel, "SEGMENTAÇÃO")
        botao(painel, "⚙ Segmentar Mama", self._segmentar).pack(padx=12, fill="x")
        self._btn_mascara = botao(painel, "👁 Ver Máscara",
                                  self._alternar_mascara,
                                  fg_color="transparent", border_width=1,
                                  state="disabled")
        self._btn_mascara.pack(padx=12, pady=4, fill="x")

        rotulo_secao(painel, "INFO")
        self._caixa_info = ctk.CTkTextbox(painel, height=130, font=FONTE_MONO,
                                          state="disabled")
        self._caixa_info.pack(padx=12, fill="x")

        # Área da imagem
        area = ctk.CTkFrame(self, corner_radius=10)
        area.pack(side="left", fill="both", expand=True)
        self._titulo_img = ctk.CTkLabel(area, text="Nenhuma imagem carregada",
                                        font=FONTE_SECAO, anchor="w")
        self._titulo_img.pack(anchor="w", padx=12, pady=(8, 4))

        fundo_canvas = ctk.CTkFrame(area, corner_radius=6)
        fundo_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._canvas = tk.Canvas(fundo_canvas, highlightthickness=0, bg="#f0f0f0")
        sv = ctk.CTkScrollbar(fundo_canvas, command=self._canvas.yview)
        sh = ctk.CTkScrollbar(fundo_canvas, orientation="horizontal",
                               command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=sv.set, xscrollcommand=sh.set)
        sh.pack(side="bottom", fill="x")
        sv.pack(side="right",  fill="y")
        self._canvas.pack(fill="both", expand=True)

    def _abrir(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")])
        if not caminho: return
        try:
            self._img_original = Image.open(caminho)
            self._img_segmentada = None
            self._mostrar_mascara = False
            self._btn_mascara.configure(state="disabled")
            self._slider_zoom.set(1.0); self._zoom = 1.0
            self._titulo_img.configure(text=os.path.basename(caminho))
            self._atualizar_info(caminho)
            self._renderizar()
            self._status.definir(f"Imagem: {os.path.basename(caminho)}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _atualizar_info(self, caminho):
        img = self._img_original
        texto = (f"Arquivo: {os.path.basename(caminho)}\n"
                 f"Tamanho: {img.width}×{img.height} px\n"
                 f"Modo:    {img.mode}\n"
                 f"Disco:   {os.path.getsize(caminho)/1024:.1f} KB")
        self._caixa_info.configure(state="normal")
        self._caixa_info.delete("1.0", "end")
        self._caixa_info.insert("1.0", texto)
        self._caixa_info.configure(state="disabled")

    def _ao_zoom(self, val):
        self._zoom = float(val)
        self._lbl_zoom.configure(text=f"{int(self._zoom*100)}%")
        self._renderizar()

    def _reset_zoom(self):
        self._slider_zoom.set(1.0); self._ao_zoom(1.0)

    def _renderizar(self):
        if not self._img_original: return
        fonte = (self._img_segmentada if self._mostrar_mascara and self._img_segmentada
                 else self._img_original)
        arr = np.array(fonte)
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if pil.mode not in ("L","RGB","RGBA"): pil = pil.convert("L")
        larg = max(1, int(pil.width  * self._zoom))
        alt  = max(1, int(pil.height * self._zoom))
        pil  = pil.resize((larg, alt), Image.LANCZOS)
        self._img_tk = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._img_tk)
        self._canvas.configure(scrollregion=(0, 0, larg, alt))

    def _segmentar(self):
        if not self._img_original:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro."); return
        self._status.definir("Segmentando…")
        threading.Thread(target=self._executar_segmentacao, daemon=True).start()

    def _executar_segmentacao(self):
        arr = np.array(self._img_original)
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
        if arr.ndim == 3: arr = arr[:, :, 0]
        # Otsu
        histograma, _ = np.histogram(arr.flatten(), 256, [0, 256])
        total = arr.size; soma_total = float(np.dot(np.arange(256), histograma))
        peso0 = soma0 = melhor_var = limiar = 0
        for t in range(256):
            peso0 += histograma[t]
            if not peso0: continue
            peso1 = total - peso0
            if not peso1: break
            soma0 += t * histograma[t]
            media0 = soma0 / peso0; media1 = (soma_total - soma0) / peso1
            var = peso0 * peso1 * (media0 - media1) ** 2
            if var > melhor_var: melhor_var = var; limiar = t
        mascara = Image.fromarray((arr > limiar).astype(np.uint8) * 255)
        mascara = mascara.filter(ImageFilter.MinFilter(5))
        mascara = mascara.filter(ImageFilter.MaxFilter(15))
        arr_mascara = np.array(mascara)
        resultado = np.where(arr_mascara > 0, arr, 0).astype(np.uint8)
        self._img_segmentada = Image.fromarray(resultado)
        self.after(0, self._pos_segmentacao)

    def _pos_segmentacao(self):
        self._mostrar_mascara = True
        self._btn_mascara.configure(state="normal", text="👁 Ver Original")
        self._renderizar()
        self._status.definir("Segmentação concluída.")

    def _alternar_mascara(self):
        self._mostrar_mascara = not self._mostrar_mascara
        self._btn_mascara.configure(
            text="👁 Ver Original" if self._mostrar_mascara else "👁 Ver Máscara")
        self._renderizar()


# ── Aba Dataset ──────────────────────────────────────────────────────────────

class AbaDataset(ctk.CTkFrame):
    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status     = status
        self._imgs_treino: list[str] = []
        self._imgs_teste:  list[str] = []
        self._construir()

    def _construir(self):
        topo = ctk.CTkFrame(self, corner_radius=10)
        topo.pack(fill="x", pady=(0, 6))
        rotulo_secao(topo, "DIRETÓRIO")
        linha = ctk.CTkFrame(topo, fg_color="transparent")
        linha.pack(fill="x", padx=12, pady=(0, 10))
        botao(linha, "📁 Selecionar Diretório", self._carregar_dir).pack(side="left")
        self._lbl_dir = ctk.CTkLabel(linha, text="—", font=FONTE_PEQUENA, anchor="w")
        self._lbl_dir.pack(side="left", padx=8)

        # Cards por classe
        frame_classes = ctk.CTkFrame(self, corner_radius=10)
        frame_classes.pack(fill="x", pady=(0, 6))
        rotulo_secao(frame_classes, "CLASSES BIRADS")
        linha_cards = ctk.CTkFrame(frame_classes, fg_color="transparent")
        linha_cards.pack(fill="x", padx=12, pady=(0, 10))
        self._cards_classe = [CartaoMetrica(linha_cards, f"BIRADS {r}")
                               for r in ("I","II","III","IV")]
        [c.pack(side="left", expand=True, fill="both", padx=3) for c in self._cards_classe]

        linha_split = ctk.CTkFrame(frame_classes, fg_color="transparent")
        linha_split.pack(fill="x", padx=12, pady=(0, 10))
        self._card_treino = CartaoMetrica(linha_split, "Treino")
        self._card_teste  = CartaoMetrica(linha_split, "Teste (múlt. 4)")
        self._card_total  = CartaoMetrica(linha_split, "Total")
        for c in (self._card_treino, self._card_teste, self._card_total):
            c.pack(side="left", expand=True, fill="both", padx=3)

        # Aumento de dados
        frame_aumento = ctk.CTkFrame(self, corner_radius=10)
        frame_aumento.pack(fill="x", pady=(0, 6))
        rotulo_secao(frame_aumento, "AUMENTO DE DADOS")
        ctk.CTkLabel(frame_aumento, font=FONTE_CORPO,
                     text="Rotações: −20° −10° 0° +10° +20°  (5× por imagem)").pack(
            anchor="w", padx=12)
        linha_aug = ctk.CTkFrame(frame_aumento, fg_color="transparent")
        linha_aug.pack(fill="x", padx=12, pady=(4, 10))
        botao(linha_aug, "⟳ Realizar Aumento", self._aumentar).pack(side="left")
        self._barra_aumento = ctk.CTkProgressBar(linha_aug)
        self._barra_aumento.set(0)
        self._barra_aumento.pack(side="left", fill="x", expand=True, padx=8)
        self._lbl_aumento = ctk.CTkLabel(linha_aug, text="0/0", font=FONTE_PEQUENA, width=50)
        self._lbl_aumento.pack(side="left")

        # Log
        frame_log = ctk.CTkFrame(self, corner_radius=10)
        frame_log.pack(fill="both", expand=True)
        rotulo_secao(frame_log, "LOG")
        self._log = ctk.CTkTextbox(frame_log, font=FONTE_MONO, state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def _registrar(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n"); self._log.see("end")
        self._log.configure(state="disabled")

    def _carregar_dir(self):
        diretorio = filedialog.askdirectory()
        if not diretorio: return
        self._lbl_dir.configure(text=diretorio)
        extensoes = {".png", ".tif", ".tiff"}
        todas = [os.path.join(r, f) for r, _, fs in os.walk(diretorio)
                 for f in sorted(fs) if os.path.splitext(f)[1].lower() in extensoes]
        if not todas:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada."); return

        mapa_classe = {"D": 0, "E": 1, "F": 2, "G": 3}
        contagem = [0, 0, 0, 0]
        treino, teste = [], []
        for caminho in todas:
            base = os.path.basename(caminho)
            digitos = "".join(c for c in os.path.splitext(base)[0] if c.isdigit())
            num = int(digitos) if digitos else 0
            (teste if num % 4 == 0 else treino).append(caminho)
            indice = mapa_classe.get(base[0].upper(), -1)
            if indice >= 0: contagem[indice] += 1

        self._imgs_treino = treino; self._imgs_teste = teste
        [self._cards_classe[i].definir(str(contagem[i])) for i in range(4)]
        self._card_treino.definir(str(len(treino)))
        self._card_teste.definir(str(len(teste)))
        self._card_total.definir(str(len(todas)))
        self._registrar(f"Diretório: {diretorio}")
        self._registrar(f"Total: {len(todas)} | Treino: {len(treino)} | Teste: {len(teste)}")
        self._status.definir(f"Dataset: {len(todas)} imagens carregadas.")

    def _aumentar(self):
        if not self._imgs_treino:
            messagebox.showwarning("Aviso", "Carregue um dataset primeiro."); return
        threading.Thread(target=self._executar_aumento, daemon=True).start()

    def _executar_aumento(self):
        angulos = [-20, -10, 0, 10, 20]
        total = len(self._imgs_treino) * len(angulos); feito = 0
        for caminho in self._imgs_treino:
            try: img = Image.open(caminho)
            except: continue
            for ang in angulos:
                img.rotate(ang, expand=False, fillcolor=0)  # salvar aqui
                feito += 1
                fracao = feito / total
                self.after(0, lambda f=fracao, d=feito, t=total: (
                    self._barra_aumento.set(f),
                    self._lbl_aumento.configure(text=f"{d}/{t}")))
                time.sleep(0.003)
        self.after(0, lambda: (
            self._registrar(f"Aumento concluído: {total} imagens geradas."),
            self._status.definir("Aumento de dados concluído.")))


# ── Aba Classificação ────────────────────────────────────────────────────────

class AbaClassificacao(ctk.CTkFrame):
    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status = status
        self._modo   = ctk.StringVar(value="binario")
        self._construir()

    def _construir(self):
        # Configuração
        cfg = ctk.CTkFrame(self, corner_radius=10)
        cfg.pack(fill="x", pady=(0, 6))
        rotulo_secao(cfg, "CLASSIFICADOR")
        linha_modo = ctk.CTkFrame(cfg, fg_color="transparent")
        linha_modo.pack(fill="x", padx=12)
        ctk.CTkRadioButton(linha_modo, text="Binário (I+II vs III+IV)",
                           variable=self._modo, value="binario",
                           font=FONTE_CORPO).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(linha_modo, text="4 Classes (I×II×III×IV)",
                           variable=self._modo, value="quadriclasse",
                           font=FONTE_CORPO).pack(side="left")
        linha_botoes = ctk.CTkFrame(cfg, fg_color="transparent")
        linha_botoes.pack(fill="x", padx=12, pady=8)
        botao(linha_botoes, "▶ Treinar",       self._treinar).pack(side="left", padx=(0,6))
        botao(linha_botoes, "⚡ Classificar",  self._classificar).pack(side="left", padx=(0,6))
        botao(linha_botoes, "💾 Salvar Modelo", self._salvar,
              fg_color="transparent", border_width=1).pack(side="left")

        linha_progresso = ctk.CTkFrame(cfg, fg_color="transparent")
        linha_progresso.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(linha_progresso, text="Épocas:", font=FONTE_PEQUENA).pack(side="left")
        self._barra_treino = ctk.CTkProgressBar(linha_progresso)
        self._barra_treino.set(0)
        self._barra_treino.pack(side="left", fill="x", expand=True, padx=6)
        self._lbl_epoca = ctk.CTkLabel(linha_progresso, text="—", font=FONTE_PEQUENA, width=50)
        self._lbl_epoca.pack(side="left")
        self._lbl_tempo = ctk.CTkLabel(linha_progresso, text="", font=FONTE_PEQUENA, width=60)
        self._lbl_tempo.pack(side="left")

        # Métricas binárias
        frame_metricas = ctk.CTkFrame(self, corner_radius=10)
        frame_metricas.pack(fill="x", pady=(0, 6))
        rotulo_secao(frame_metricas, "MÉTRICAS – BINÁRIO")
        linha_metricas = ctk.CTkFrame(frame_metricas, fg_color="transparent")
        linha_metricas.pack(fill="x", padx=12, pady=(0, 10))
        nomes = ["Sensib.", "Especif.", "Precisão", "Acurácia", "F1"]
        self._cards_metrica = [CartaoMetrica(linha_metricas, n) for n in nomes]
        [c.pack(side="left", expand=True, fill="both", padx=3)
         for c in self._cards_metrica]

        # Matriz de confusão
        frame_matriz = ctk.CTkFrame(self, corner_radius=10)
        frame_matriz.pack(fill="both", expand=True, pady=(0, 0))
        rotulo_secao(frame_matriz, "MATRIZ DE CONFUSÃO – 4 CLASSES")
        area_matriz = ctk.CTkFrame(frame_matriz, fg_color="transparent")
        area_matriz.pack(padx=12, pady=(0, 8))
        rotulos = ["I", "II", "III", "IV"]
        linha_cab = ctk.CTkFrame(area_matriz, fg_color="transparent")
        linha_cab.pack()
        ctk.CTkLabel(linha_cab, text="Pred→\nReal↓", font=FONTE_PEQUENA, width=70).pack(side="left")
        for r in rotulos:
            ctk.CTkLabel(linha_cab, text=f"B{r}", font=FONTE_PEQUENA, width=70).pack(side="left")
        self._celulas_cm: list[list[ctk.CTkLabel]] = []
        for i, ri in enumerate(rotulos):
            linha_cm = ctk.CTkFrame(area_matriz, fg_color="transparent")
            linha_cm.pack(pady=1)
            ctk.CTkLabel(linha_cm, text=f"B{ri}", font=FONTE_PEQUENA, width=70).pack(side="left")
            linha_cel = []
            for j in range(4):
                cel = ctk.CTkLabel(linha_cm, text="—", font=FONTE_CORPO,
                                   corner_radius=4, width=66, height=30,
                                   fg_color=("#d0e8ff" if i == j else "#f5f5f5"))
                cel.pack(side="left", padx=2)
                linha_cel.append(cel)
            self._celulas_cm.append(linha_cel)

        linha_m4 = ctk.CTkFrame(frame_matriz, fg_color="transparent")
        linha_m4.pack(fill="x", padx=12, pady=(4, 10))
        self._card_sens_media  = CartaoMetrica(linha_m4, "Sensib. Média")
        self._card_espec_media = CartaoMetrica(linha_m4, "Especif. Média")
        self._card_tempo_exec  = CartaoMetrica(linha_m4, "Tempo")
        for c in (self._card_sens_media, self._card_espec_media, self._card_tempo_exec):
            c.pack(side="left", expand=True, fill="both", padx=3)

    def _treinar(self):
        threading.Thread(target=self._executar_treino, daemon=True).start()

    def _executar_treino(self):
        epocas = 20; t0 = time.time()
        self.after(0, lambda: self._status.definir("Treinando modelo…"))
        for ep in range(1, epocas + 1):
            time.sleep(0.1)
            fracao = ep / epocas; decorrido = time.time() - t0
            self.after(0, lambda f=fracao, e=ep, t=decorrido: (
                self._barra_treino.set(f),
                self._lbl_epoca.configure(text=f"{e}/{epocas}"),
                self._lbl_tempo.configure(text=f"{t:.1f}s")))
        self.after(0, lambda: self._status.definir("Treino concluído."))

    def _classificar(self):
        t0 = time.time()
        sens = random.uniform(0.72, 0.92); espec = random.uniform(0.74, 0.94)
        prec = random.uniform(0.70, 0.90); acc   = random.uniform(0.75, 0.93)
        f1   = 2 * prec * sens / (prec + sens)
        decorrido = time.time() - t0 + random.uniform(0.5, 3)
        for cartao, v in zip(self._cards_metrica, [sens, espec, prec, acc, f1]):
            cartao.definir(f"{v:.3f}")
        matriz = np.random.randint(0, 40, (4, 4))
        np.fill_diagonal(matriz, np.random.randint(60, 120, 4))
        for i in range(4):
            for j in range(4):
                self._celulas_cm[i][j].configure(text=str(matriz[i, j]))
        self._card_sens_media.definir(f"{sens:.3f}")
        self._card_espec_media.definir(f"{espec:.3f}")
        self._card_tempo_exec.definir(f"{decorrido:.2f}s")
        self._status.definir("Classificação concluída.")

    def _salvar(self):
        caminho = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch", "*.pth"), ("H5", "*.h5"), ("Todos", "*.*")])
        if caminho:
            messagebox.showinfo("Salvar", f"Implemente torch.save() aqui:\n{caminho}")
            self._status.definir(f"Modelo: {os.path.basename(caminho)}")


# ── Aba Grad-CAM ─────────────────────────────────────────────────────────────

class AbaGradCAM(ctk.CTkFrame):
    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status    = status
        self._caminho   = ""
        self._pil_orig  = None
        self._construir()

    def _construir(self):
        painel = ctk.CTkFrame(self, width=220, corner_radius=10)
        painel.pack(side="left", fill="y", padx=(0, 6))
        painel.pack_propagate(False)
        rotulo_secao(painel, "IMAGEM")
        botao(painel, "📂 Abrir Imagem", self._abrir).pack(padx=12, fill="x")
        self._lbl_nome = ctk.CTkLabel(painel, text="—", font=FONTE_PEQUENA,
                                      wraplength=200, anchor="w")
        self._lbl_nome.pack(padx=12, pady=4, anchor="w")
        rotulo_secao(painel, "RESULTADO")
        self._card_classe     = CartaoMetrica(painel, "BIRADS Predito")
        self._card_confianca  = CartaoMetrica(painel, "Confiança")
        self._card_classe.pack(padx=12, fill="x")
        self._card_confianca.pack(padx=12, fill="x", pady=6)
        botao(painel, "🔥 Gerar Grad-CAM", self._gerar).pack(padx=12, fill="x")
        rotulo_secao(painel, "LEGENDA")
        ctk.CTkLabel(painel, text="Azul→baixo  Verde→médio\nAmarelo→alto  Vermelho→máx",
                     font=FONTE_PEQUENA, justify="left", anchor="w").pack(padx=12, anchor="w")

        area = ctk.CTkFrame(self, corner_radius=10)
        area.pack(side="left", fill="both", expand=True)
        cabecalho = ctk.CTkFrame(area, fg_color="transparent", height=30)
        cabecalho.pack(fill="x", padx=12, pady=(8, 4)); cabecalho.pack_propagate(False)
        ctk.CTkLabel(cabecalho, text="Original",    font=FONTE_SECAO, anchor="w").pack(side="left", expand=True)
        ctk.CTkLabel(cabecalho, text="Grad-CAM", font=FONTE_SECAO, anchor="w").pack(side="left", expand=True)
        area_canvas = ctk.CTkFrame(area, corner_radius=6, fg_color="#e8e8e8")
        area_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._canvas_orig = tk.Canvas(area_canvas, bg="#e8e8e8", highlightthickness=0)
        self._canvas_cam  = tk.Canvas(area_canvas, bg="#e8e8e8", highlightthickness=0)
        self._canvas_orig.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self._canvas_cam.pack( side="left", fill="both", expand=True, padx=(2, 0))

    def _abrir(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")])
        if not caminho: return
        self._caminho = caminho
        self._lbl_nome.configure(text=os.path.basename(caminho))
        try:
            img = Image.open(caminho)
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
            self._pil_orig = Image.fromarray(arr).convert("L")
            self._exibir_canvas(self._canvas_orig, self._pil_orig, "_tk_orig")
            self._status.definir(f"Imagem: {os.path.basename(caminho)}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _exibir_canvas(self, canvas, img, attr):
        canvas.update_idletasks()
        larg = max(canvas.winfo_width(),  200)
        alt  = max(canvas.winfo_height(), 200)
        copia = img.copy(); copia.thumbnail((larg, alt), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(copia)
        setattr(self, attr, img_tk)
        canvas.delete("all")
        canvas.create_image(larg // 2, alt // 2, anchor="center", image=img_tk)

    def _gerar(self):
        if not self._caminho:
            messagebox.showwarning("Aviso", "Selecione uma imagem."); return
        threading.Thread(target=self._executar_gradcam, daemon=True).start()

    def _executar_gradcam(self):
        self.after(0, lambda: self._status.definir("Gerando Grad-CAM…"))
        time.sleep(0.4)
        orig = self._pil_orig.copy(); larg, alt = orig.size
        # Heatmap gaussiano sintético (substituir por gradientes reais)
        cx, cy = larg * random.uniform(0.3, 0.7), alt * random.uniform(0.3, 0.7)
        sigma2 = (min(larg, alt) * 0.2) ** 2
        xs, ys = np.meshgrid(np.arange(larg), np.arange(alt))
        mapa_calor = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma2)).astype(np.float32)
        mapa_pil   = Image.fromarray((mapa_calor * 255).astype(np.uint8))
        mapa_pil   = mapa_pil.filter(ImageFilter.GaussianBlur(max(larg, alt) // 20))
        mapa_arr   = np.array(mapa_pil) / 255.0
        # Colormap jet vetorizado
        r = np.clip(1.5 - np.abs(4 * mapa_arr - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * mapa_arr - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * mapa_arr - 1), 0, 1)
        cam_rgb  = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
        cam_pil  = Image.fromarray(cam_rgb)
        misturado = Image.blend(orig.convert("RGB"), cam_pil, alpha=0.55)
        birads     = random.choice(["I", "II", "III", "IV"])
        confianca  = random.uniform(0.70, 0.99)
        self.after(0, lambda m=misturado, b=birads, c=confianca:
                   self._exibir_resultado(m, b, c))

    def _exibir_resultado(self, misturado, birads, confianca):
        self._exibir_canvas(self._canvas_cam, misturado, "_tk_cam")
        self._card_classe.definir(f"BIRADS {birads}")
        self._card_confianca.definir(f"{confianca:.1%}")
        self._status.definir("Grad-CAM gerado.")


# ── Aplicação principal ──────────────────────────────────────────────────────

class AplicacaoMamografia(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MamoVision — Segmentação e Classificação Mamográfica")
        self.geometry("1280x800"); self.minsize(900, 640)
        self._construir()

    def _construir(self):
        cabecalho = ctk.CTkFrame(self, corner_radius=0, height=52)
        cabecalho.pack(fill="x"); cabecalho.pack_propagate(False)
        ctk.CTkLabel(cabecalho, text="MamoVision",
                     font=FONTE_TITULO).pack(side="left", padx=12)
        ctk.CTkLabel(cabecalho,
                     text="Segmentação e Classificação Mamográfica · PUC Minas",
                     font=FONTE_PEQUENA).pack(side="left")

        self._barra_status = BarraStatus(self)
        self._barra_status.pack(fill="x", side="bottom")

        abas = ctk.CTkTabview(self)
        abas.pack(fill="both", expand=True, padx=10, pady=6)
        nomes_abas = ["📷 Visualizador", "📦 Dataset", "🧠 Classificação", "🔥 Grad-CAM"]
        classes_abas = [AbaVisualizador, AbaDataset, AbaClassificacao, AbaGradCAM]
        for nome, cls in zip(nomes_abas, classes_abas):
            abas.add(nome)
            cls(abas.tab(nome), self._barra_status).pack(fill="both", expand=True)


def main():
    app = AplicacaoMamografia()
    app.mainloop()


if __name__ == "__main__":
    main()
