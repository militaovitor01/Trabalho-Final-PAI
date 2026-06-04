# =============================================================================
# Segmentação e Classificação de Imagens Mamográficas
# Disciplina: Processamento e Análise de Imagens - PUC Minas
# Prof. Alexei Machado
# Grupo: [NOME, MATRÍCULA, CURSO E CAMPUS DOS INTEGRANTES]
# =============================================================================

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading, time, random, math, os, re, shutil
from PIL import Image, ImageTk, ImageFilter
import numpy as np

# ── Bibliotecas de processamento e Deep Learning ────────────────────────────
# scipy é usada para encontrar componentes conectados na segmentação
try:
    from scipy import ndimage as ndi
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# PyTorch e torchvision para DataLoaders e transformações
try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

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
        # Chama o pipeline robusto de segmentação e armazena o resultado
        img_seg = segmentar_mama(self._img_original)
        self._img_segmentada = img_seg
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


# =============================================================================
# PIPELINE DE PREPARAÇÃO DE DADOS
# Funções independentes da UI, reutilizáveis pelas abas de Classificação e
# Grad-CAM quando necessário.
# =============================================================================


def organizar_registro(caminho_arquivo: str) -> dict | None:
    """
    Extrai metadados de uma imagem a partir do seu caminho no dataset LMLO.

    Regras do professor:
    - A letra inicial do nome do arquivo define a classe:
        D → BI-RADS I  (classe 0)
        E → BI-RADS II (classe 1)
        F → BI-RADS III(classe 2)
        G → BI-RADS IV (classe 3)
    - Imagens cujo número (dígitos do nome) seja múltiplo de 4 → teste
    - Demais → treino

    Retorna um dicionário com todos os metadados ou None se o arquivo
    não pertencer a nenhuma classe reconhecida.

    Exemplo de retorno:
        {
            "arquivo":  "/caminho/LMLO/D+left+MLO/D001.png",
            "letra":    "D",
            "classe":   0,
            "birads":   "I",
            "numero":   1,
            "treino":   True
        }
    """
    mapa = {"D": (0, "I"), "E": (1, "II"), "F": (2, "III"), "G": (3, "IV")}

    nome  = os.path.basename(caminho_arquivo)
    letra = nome[0].upper() if nome else ""

    if letra not in mapa:
        return None  # arquivo sem prefixo de classe reconhecível

    classe, birads = mapa[letra]

    # Extrai apenas os dígitos do nome (sem extensão) para determinar o número
    nome_sem_ext = os.path.splitext(nome)[0]
    digitos = re.sub(r"\D", "", nome_sem_ext)
    numero  = int(digitos) if digitos else 0

    # Regra de split: múltiplo de 4 → teste; caso contrário → treino
    eh_treino = (numero % 4 != 0)

    return {
        "arquivo": caminho_arquivo,
        "letra":   letra,
        "classe":  classe,
        "birads":  birads,
        "numero":  numero,
        "treino":  eh_treino,
    }


def limiar_otsu(arr_cinza: np.ndarray) -> int:
    """
    Calcula o limiar ótimo de Otsu para uma imagem em escala de cinza (uint8).

    O método maximiza a variância inter-classes, separando o fundo escuro
    (fundo preto das mamografias) do tecido mamário mais claro.

    Parâmetros:
        arr_cinza: array 2D uint8 com a imagem em escala de cinza.

    Retorna:
        limiar (int): valor de intensidade [0,255] que maximiza a variância.
    """
    histograma, _ = np.histogram(arr_cinza.flatten(), bins=256, range=(0, 256))
    total         = arr_cinza.size
    soma_total    = float(np.dot(np.arange(256), histograma))

    peso0 = soma0 = 0
    melhor_var = limiar = 0

    for t in range(256):
        peso0 += histograma[t]
        if not peso0:
            continue
        peso1 = total - peso0
        if not peso1:
            break
        soma0 += t * histograma[t]
        media0 = soma0 / peso0
        media1 = (soma_total - soma0) / peso1
        # Variância inter-classes: produto dos pesos × quadrado da diferença de médias
        var = peso0 * peso1 * (media0 - media1) ** 2
        if var > melhor_var:
            melhor_var = var
            limiar     = t

    return limiar


def maior_componente_conectado(mascara_bin: np.ndarray) -> np.ndarray:
    """
    Retorna uma máscara binária contendo apenas o maior componente conectado.

    Em mamografias, após a limiarização de Otsu pode haver pequenos artefatos
    (anotações, ruídos de borda) que formam componentes separados do tecido
    mamário principal.  Ao reter apenas o maior componente isolamos a mama.

    Parâmetros:
        mascara_bin: array 2D booleano/uint8 (True/1 = objeto, False/0 = fundo).

    Retorna:
        Array 2D uint8 com 255 onde está o maior componente e 0 no resto.
    """
    if SCIPY_OK:
        # Usa scipy.ndimage para rotular componentes conectados com
        # conectividade total (estrutura 3×3 = 8-conectividade)
        estrutura = ndi.generate_binary_structure(2, 2)
        rotulado, n_comp = ndi.label(mascara_bin, structure=estrutura)
        if n_comp == 0:
            return mascara_bin.astype(np.uint8) * 255
        # Conta pixels por componente (ignora rótulo 0 = fundo)
        tamanhos = ndi.sum(mascara_bin, rotulado, range(1, n_comp + 1))
        maior    = int(np.argmax(tamanhos)) + 1
        return (rotulado == maior).astype(np.uint8) * 255
    else:
        # Fallback sem scipy: usa erosão/dilatação PIL para suprimir pequenos artefatos
        pil = Image.fromarray(mascara_bin.astype(np.uint8) * 255)
        pil = pil.filter(ImageFilter.MinFilter(9))   # erosão forte remove artefatos pequenos
        pil = pil.filter(ImageFilter.MaxFilter(25))  # dilatação restaura a mama principal
        return np.array(pil)


def segmentar_mama(imagem_pil: Image.Image) -> Image.Image:
    """
    Pipeline robusto de segmentação da mama em imagem mamográfica.

    Etapas:
      1. Converte para escala de cinza normalizada (uint8 0-255)
      2. Aplica limiarização de Otsu para binarizar fundo vs. tecido
      3. Seleciona apenas o maior componente conectado (remove artefatos)
      4. Remove ruídos residuais via erosão seguida de dilatação (abertura morfológica)
      5. Aplica a máscara à imagem original (fundo → 0)

    Parâmetros:
        imagem_pil: imagem PIL em qualquer modo e profundidade de bits.

    Retorna:
        Imagem PIL em modo 'L' (escala de cinza) com fundo zerado e mama isolada.
    """
    # 1. Escala de cinza normalizada para uint8
    arr = np.array(imagem_pil)
    if arr.dtype != np.uint8:
        # Normaliza qualquer profundidade (8, 12, 16 bits) para 0-255
        mn, mx = arr.min(), arr.max()
        arr = ((arr.astype(np.float32) - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
    if arr.ndim == 3:
        # Converte RGB/RGBA → cinza usando médias dos canais (canal 0 para mamografias)
        arr = arr[:, :, 0]

    # 2. Limiarização de Otsu — separa fundo preto do tecido mamário
    limiar   = limiar_otsu(arr)
    mascara  = (arr > limiar).astype(np.uint8)

    # 3. Maior componente conectado — elimina artefatos externos (anotações, réguas)
    mascara = maior_componente_conectado(mascara)

    # 4. Remoção de ruídos:
    #    - MinFilter(5): erosão leve (remove ruídos de 1-2px na borda)
    #    - MaxFilter(9): dilatação para recuperar a borda da mama
    # Tamanho 5 e 9 foram escolhidos empiricamente para mamografias LMLO de ~2000px.
    # Para imagens menores o efeito é proporcional pois PIL usa kernel absoluto —
    # mas as imagens do dataset têm resoluções similares entre si.
    pil_mascara = Image.fromarray(mascara)
    pil_mascara = pil_mascara.filter(ImageFilter.MinFilter(5))
    pil_mascara = pil_mascara.filter(ImageFilter.MaxFilter(9))
    arr_mascara = np.array(pil_mascara)

    # 5. Aplica a máscara: mantém pixels da mama, zera o fundo
    resultado = np.where(arr_mascara > 0, arr, 0).astype(np.uint8)
    return Image.fromarray(resultado)


def recortar_bounding_box(imagem_seg: Image.Image) -> Image.Image:
    """
    Recorta a região útil da mama após a segmentação.

    Após a segmentação, a imagem ainda contém grandes áreas de fundo preto
    que ocupam espaço desnecessário e degradam a qualidade do treinamento
    (a rede desperdiça capacidade modelando pixels pretos).

    O recorte encontra o menor retângulo envolvente (bounding box) dos pixels
    não-nulos e retorna apenas essa região.

    Se a mama ocupa 30% da imagem, após o recorte ocupa ~100%.

    Parâmetros:
        imagem_seg: imagem PIL em modo 'L' já segmentada.

    Retorna:
        Imagem PIL recortada ou a imagem original se não houver pixels válidos.
    """
    arr = np.array(imagem_seg)
    # Encontra linhas e colunas que contenham ao menos um pixel não-zero
    linhas_validas = np.any(arr > 0, axis=1)
    cols_validas   = np.any(arr > 0, axis=0)

    if not linhas_validas.any():
        return imagem_seg  # imagem completamente preta: retorna sem modificar

    lin_min, lin_max = np.where(linhas_validas)[0][[0, -1]]
    col_min, col_max = np.where(cols_validas)[0][[0, -1]]

    # Margem de 2px para não cortar a borda do tecido mamário
    lin_min = max(0, lin_min - 2)
    lin_max = min(arr.shape[0] - 1, lin_max + 2)
    col_min = max(0, col_min - 2)
    col_max = min(arr.shape[1] - 1, col_max + 2)

    arr_crop = arr[lin_min:lin_max + 1, col_min:col_max + 1]
    return Image.fromarray(arr_crop)


def preparar_imagem(imagem_pil: Image.Image) -> Image.Image:
    """
    Pipeline completo de preparação de uma imagem para as redes DenseNet121/VGG16.

    Etapas:
      1. Segmentação robusta da mama (Otsu + componente conectado + morfologia)
      2. Recorte do bounding box (elimina grandes áreas vazias de fundo)
      3. Redimensionamento para 224×224 px (tamanho exigido pelas redes pré-treinadas)
      4. Conversão para RGB (3 canais), replicando o canal cinza — as redes
         ImageNet esperam 3 canais mas a informação diagnóstica é monocromática.

    Parâmetros:
        imagem_pil: imagem PIL original (qualquer modo/profundidade).

    Retorna:
        Imagem PIL 224×224 RGB pronta para ser processada pela rede.
    """
    # Etapa 1: segmentação
    img_seg = segmentar_mama(imagem_pil)

    # Etapa 2: recorte da região útil
    img_crop = recortar_bounding_box(img_seg)

    # Etapa 3: redimensionamento para 224×224 (DenseNet/VGG padrão ImageNet)
    # LANCZOS oferece melhor qualidade para downscaling de imagens de alta resolução
    img_224 = img_crop.resize((224, 224), Image.LANCZOS)

    # Etapa 4: conversão para RGB — replica o canal cinza em R, G e B
    # Isso é necessário porque os pesos ImageNet esperam 3 canais.
    img_rgb = img_224.convert("RGB")

    return img_rgb


def criar_dataloaders(
    dir_processado: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """
    Cria DataLoaders PyTorch a partir da estrutura processed/ gerada pelo pipeline.

    Utiliza torchvision.datasets.ImageFolder, que lê automaticamente subpastas
    como classes (D, E, F, G) e associa os rótulos corretos.

    Transformações aplicadas:
    - Treino:
        • ToTensor()       — converte PIL → tensor [0,1]
        • Normalize(mean, std) — normalização ImageNet (usada pelo DenseNet/VGG)
    - Teste:
        • ToTensor() + Normalize() (sem augmentation para avaliação justa)

    As médias e desvios-padrão são os valores canônicos do ImageNet:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    Usar esses valores é correto pois as redes foram pré-treinadas com eles.

    Parâmetros:
        dir_processado: caminho para processed/ (deve conter subpastas train/ e test/)
        batch_size:     número de amostras por batch (padrão 32 — bom equilíbrio
                        entre velocidade e estabilidade do gradiente)
        num_workers:    workers para carregamento paralelo (0 = thread principal,
                        evita problemas no Windows com multiprocessing)

    Retorna:
        (train_loader, test_loader) — tupla de DataLoaders ou (None, None) se
        os diretórios não existirem.
    """
    if not TORCH_OK:
        return None, None

    # Normalização ImageNet (valores canônicos para redes pré-treinadas)
    media_imagenet  = [0.485, 0.456, 0.406]
    desvio_imagenet = [0.229, 0.224, 0.225]

    # Transformação para treino: converte para tensor e normaliza
    # (data augmentation já foi aplicada e salva em disco na etapa anterior)
    transf_treino = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=media_imagenet, std=desvio_imagenet),
    ])

    # Transformação para teste: idêntica ao treino, sem augmentation
    transf_teste = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=media_imagenet, std=desvio_imagenet),
    ])

    dir_treino = os.path.join(dir_processado, "train")
    dir_teste  = os.path.join(dir_processado, "test")

    train_loader = test_loader = None

    if os.path.isdir(dir_treino):
        # ImageFolder espera: dir_treino/D/*.png, dir_treino/E/*.png, ...
        dataset_treino = datasets.ImageFolder(dir_treino, transform=transf_treino)
        train_loader   = DataLoader(
            dataset_treino,
            batch_size=batch_size,
            shuffle=True,       # embaralha a cada época para evitar overfitting
            num_workers=num_workers,
            pin_memory=False,   # pin_memory=True apenas se GPU disponível
        )

    if os.path.isdir(dir_teste):
        dataset_teste = datasets.ImageFolder(dir_teste, transform=transf_teste)
        test_loader   = DataLoader(
            dataset_teste,
            batch_size=batch_size,
            shuffle=False,      # não embaralha: métricas devem ser reproduzíveis
            num_workers=num_workers,
            pin_memory=False,
        )

    return train_loader, test_loader


# ── Aba Dataset ──────────────────────────────────────────────────────────────

class AbaDataset(ctk.CTkFrame):
    """
    Aba responsável por toda a etapa de preparação dos dados:
    - Leitura e organização automática do dataset LMLO
    - Segmentação, recorte e redimensionamento das imagens
    - Data Augmentation por rotação (somente treino)
    - Geração da estrutura processed/ para uso pelas redes
    - Criação de DataLoaders PyTorch prontos para treinamento
    """

    # Mapeamento letra-inicial → (índice de classe, BI-RADS)
    MAPA_CLASSE = {"D": (0, "I"), "E": (1, "II"), "F": (2, "III"), "G": (3, "IV")}
    # Ângulos de rotação para data augmentation
    ANGULOS_AUG = [-20, -10, 0, 10, 20]
    # Tamanho-alvo exigido pela DenseNet121 e VGG16
    TAMANHO_ALVO = (224, 224)

    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status          = status
        # Lista de dicionários com metadados de cada imagem do dataset
        self._registros: list[dict] = []
        # Listas de caminhos separados por split
        self._imgs_treino: list[str] = []
        self._imgs_teste:  list[str] = []
        # Diretório raiz do dataset e do diretório processado
        self._dir_dataset  = ""
        self._dir_processado = ""
        # DataLoaders (criados após o processamento)
        self._train_loader = None
        self._test_loader  = None
        self._construir()

    # ── Construção da UI (inalterada visualmente) ────────────────────────────

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
                               for r in ("I", "II", "III", "IV")]
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

    # ── Utilitário de log thread-safe ────────────────────────────────────────

    def _registrar(self, msg):
        """Insere uma linha no log de forma segura a partir de qualquer thread."""
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_ts(self, msg):
        """Agenda _registrar na thread principal (seguro para uso em threads)."""
        self.after(0, lambda m=msg: self._registrar(m))

    # ── BOTÃO 1: Selecionar Diretório ────────────────────────────────────────

    def _carregar_dir(self):
        """
        Abre diálogo para selecionar o diretório raiz do dataset LMLO.
        Percorre todas as subpastas, identifica cada imagem, extrai:
          - classe (letra inicial D/E/F/G)
          - número da imagem (para divisão treino/teste)
          - split (treino se num % 4 != 0, teste se num % 4 == 0)
        Armazena os metadados em self._registros e atualiza os cards da UI.
        Depois processa as imagens (segmentação + crop + redimensionamento)
        e gera a estrutura processed/ em disco.
        """
        diretorio = filedialog.askdirectory()
        if not diretorio:
            return
        self._dir_dataset = diretorio
        self._lbl_dir.configure(text=diretorio)

        # Extensões aceitas (PNG e TIFF conforme especificação)
        extensoes_validas = {".png", ".tif", ".tiff"}

        # Coleta todos os arquivos de imagem recursivamente e ordena pelo nome
        todos_caminhos = []
        for raiz, _, arquivos in os.walk(diretorio):
            for arq in sorted(arquivos):
                if os.path.splitext(arq)[1].lower() in extensoes_validas:
                    todos_caminhos.append(os.path.join(raiz, arq))

        if not todos_caminhos:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada."); return

        # --- Organização automática do dataset ---
        self._registros = []
        contagem_classe = [0, 0, 0, 0]
        treino, teste   = [], []

        for caminho in todos_caminhos:
            rec = organizar_registro(caminho)
            if rec is None:
                continue  # arquivo sem prefixo reconhecível: ignora
            self._registros.append(rec)
            contagem_classe[rec["classe"]] += 1
            if rec["treino"]:
                treino.append(caminho)
            else:
                teste.append(caminho)

        self._imgs_treino = treino
        self._imgs_teste  = teste

        # Atualiza cards
        for i in range(4):
            self._cards_classe[i].definir(str(contagem_classe[i]))
        self._card_treino.definir(str(len(treino)))
        self._card_teste.definir(str(len(teste)))
        self._card_total.definir(str(len(todos_caminhos)))

        self._registrar(f"Diretório: {diretorio}")
        self._registrar(f"Total: {len(todos_caminhos)} imagens  |  "
                        f"Treino: {len(treino)}  |  Teste: {len(teste)}")
        for letra, (idx, birads) in self.MAPA_CLASSE.items():
            self._registrar(f"  BI-RADS {birads} ({letra}): {contagem_classe[idx]} imagens")
        self._status.definir(f"Dataset: {len(todos_caminhos)} imagens carregadas.")

        # Inicia processamento (segmentação + crop + resize + cópia para processed/)
        # em thread separada para não travar a UI
        threading.Thread(target=self._executar_processamento, daemon=True).start()

    # ── Pipeline de processamento (thread) ───────────────────────────────────

    def _executar_processamento(self):
        """
        Para cada imagem do dataset:
          1. Abre a imagem original
          2. Aplica segmentação robusta (Otsu + maior componente + ruído)
          3. Recorta o bounding box da mama (elimina áreas vazias)
          4. Redimensiona para 224×224
          5. Converte para RGB (3 canais, exigido pela DenseNet/VGG)
          6. Salva em processed/<split>/<letra_classe>/<nome>.png
        """
        total  = len(self._registros)
        if total == 0:
            return

        # Cria a estrutura de diretórios processed/
        self._dir_processado = os.path.join(self._dir_dataset, "processed")
        for split in ("train", "test"):
            for letra in self.MAPA_CLASSE:
                os.makedirs(os.path.join(self._dir_processado, split, letra),
                            exist_ok=True)

        self._log_ts("Iniciando processamento das imagens (segmentação + crop + resize)…")
        self.after(0, lambda: self._status.definir("Processando imagens…"))

        for i, rec in enumerate(self._registros):
            caminho  = rec["arquivo"]
            letra    = rec["letra"]
            split    = "train" if rec["treino"] else "test"
            nome_arq = os.path.basename(caminho)
            destino  = os.path.join(self._dir_processado, split, letra, nome_arq)

            try:
                img = Image.open(caminho)
                # Aplica o pipeline completo de preparação
                img_proc = preparar_imagem(img)
                img_proc.save(destino)
            except Exception as e:
                self._log_ts(f"  [ERRO] {nome_arq}: {e}")

            # Atualiza progresso na UI a cada 10 imagens ou na última
            if (i + 1) % 10 == 0 or (i + 1) == total:
                self._log_ts(f"  Processadas: {i+1}/{total}")
                self.after(0, lambda v=(i+1)/total: self._barra_aumento.set(v))

        # Cria os DataLoaders após processar todas as imagens
        if TORCH_OK:
            self._train_loader, self._test_loader = criar_dataloaders(
                self._dir_processado)
            n_tr = len(self._train_loader.dataset) if self._train_loader else 0
            n_te = len(self._test_loader.dataset)  if self._test_loader  else 0
            self._log_ts(f"DataLoaders criados: train={n_tr} amostras, test={n_te} amostras")
        else:
            self._log_ts("PyTorch não encontrado — DataLoaders não criados.")

        self.after(0, lambda: (
            self._barra_aumento.set(1.0),
            self._status.definir("Processamento concluído.")))
        self._log_ts(f"Estrutura salva em: {self._dir_processado}")

    # ── BOTÃO 2: Realizar Aumento ────────────────────────────────────────────

    def _aumentar(self):
        """Valida pré-condições e lança o data augmentation em thread separada."""
        if not self._imgs_treino:
            messagebox.showwarning("Aviso", "Carregue um dataset primeiro.")
            return
        if not self._dir_processado or not os.path.isdir(self._dir_processado):
            messagebox.showwarning(
                "Aviso",
                "Aguarde o processamento das imagens ser concluído antes de aumentar.")
            return
        threading.Thread(target=self._executar_aumento, daemon=True).start()

    def _executar_aumento(self):
        """
        Data Augmentation real — apenas para o conjunto de treino.

        Para cada imagem de treino já processada (224×224 RGB) gera 5 versões
        rotacionadas em -20°, -10°, 0°, +10° e +20°, salvando cada uma como:
            <nome_original>_rot_m20.png
            <nome_original>_rot_m10.png
            <nome_original>_rot_0.png
            <nome_original>_rot_p10.png
            <nome_original>_rot_p20.png

        Sufixo adotado: m = minus (negativo), p = plus (positivo).
        A rotação é feita com fundo preto (fillcolor=0) para não inserir
        artefatos de borda nas imagens mamográficas.
        """
        # Mapeia sufixo de arquivo para cada ângulo
        sufixos = {-20: "m20", -10: "m10", 0: "0", 10: "p10", 20: "p20"}

        # Conta apenas imagens de treino já processadas
        registros_treino = [r for r in self._registros if r["treino"]]
        total   = len(registros_treino) * len(self.ANGULOS_AUG)
        feito   = 0
        geradas = 0

        self._log_ts(f"Aumento de dados: {len(registros_treino)} imagens × "
                     f"{len(self.ANGULOS_AUG)} rotações = {total} arquivos")
        self.after(0, lambda: self._barra_aumento.set(0))
        self.after(0, lambda: self._status.definir("Realizando aumento de dados…"))

        for rec in registros_treino:
            letra   = rec["letra"]
            nome_arq = os.path.basename(rec["arquivo"])
            nome_sem_ext = os.path.splitext(nome_arq)[0]

            # Lê a versão já processada (224×224 RGB) do diretório processed/
            caminho_proc = os.path.join(
                self._dir_processado, "train", letra, nome_arq)
            if not os.path.isfile(caminho_proc):
                # Se ainda não foi processada, processa agora inline
                try:
                    img_proc = preparar_imagem(Image.open(rec["arquivo"]))
                except Exception as e:
                    self._log_ts(f"  [ERRO ao abrir] {nome_arq}: {e}")
                    feito += len(self.ANGULOS_AUG)
                    continue
            else:
                try:
                    img_proc = Image.open(caminho_proc)
                except Exception as e:
                    self._log_ts(f"  [ERRO ao ler processada] {nome_arq}: {e}")
                    feito += len(self.ANGULOS_AUG)
                    continue

            # Gera as 5 rotações e salva em disco
            for ang in self.ANGULOS_AUG:
                suf  = sufixos[ang]
                nome_aug = f"{nome_sem_ext}_rot_{suf}.png"
                destino  = os.path.join(
                    self._dir_processado, "train", letra, nome_aug)

                # Rotação com PIL: expand=False mantém 224×224,
                # fillcolor=0 preenche bordas com preto (fundo padrão das mamografias)
                img_rot = img_proc.rotate(
                    ang, expand=False, fillcolor=(0, 0, 0), resample=Image.BILINEAR)
                img_rot.save(destino)
                geradas += 1
                feito   += 1

                # Atualiza barra de progresso
                fracao = feito / total
                self.after(0, lambda f=fracao, d=feito, t=total: (
                    self._barra_aumento.set(f),
                    self._lbl_aumento.configure(text=f"{d}/{t}")))

        self.after(0, lambda: (
            self._registrar(f"Aumento concluído: {geradas} imagens geradas em "
                            f"{os.path.join(self._dir_processado, 'train')}"),
            self._status.definir("Aumento de dados concluído.")))

        # Recria os DataLoaders para incluir as imagens aumentadas
        if TORCH_OK and self._dir_processado:
            self._train_loader, self._test_loader = criar_dataloaders(
                self._dir_processado)
            n_tr = len(self._train_loader.dataset) if self._train_loader else 0
            self._log_ts(f"DataLoaders recriados: train={n_tr} amostras (com augmentation)")


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