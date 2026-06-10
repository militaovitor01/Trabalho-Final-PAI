# =============================================================================
# Segmentação e Classificação de Imagens Mamográficas
# Disciplina: Processamento e Análise de Imagens - PUC Minas
# Prof. Alexei Machado
# Grupo: [NOME, MATRÍCULA, CURSO E CAMPUS DOS INTEGRANTES]
# =============================================================================

import io
import json
import os
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageFilter, ImageTk

# ── Bibliotecas de processamento e Deep Learning ────────────────────────────
try:
    from scipy import ndimage as ndi

    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import matplotlib

    matplotlib.use("Agg")  # backend sem janela (renderiza para buffer)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    MPL_OK = True
except ImportError:
    MPL_OK = False

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# =============================================================================
# SISTEMA DE LOGGING — saída simultânea para terminal (stdout) e interface
# =============================================================================
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
_logger = logging.getLogger("MamoVision")


def log_info(msg: str):
    """Emite log INFO no terminal e retorna a string formatada para a UI."""
    _logger.info(msg)
    return f"[INFO] {msg}"


def log_warn(msg: str):
    _logger.warning(msg)
    return f"[WARN] {msg}"


def log_erro(msg: str):
    _logger.error(msg)
    return f"[ERRO] {msg}"

# ── Constantes de fonte ──────────────────────────────────────────────────────
FONTE_TITULO = ("Helvetica", 16, "bold")
FONTE_SECAO = ("Helvetica", 11, "bold")
FONTE_CORPO = ("Helvetica", 11)
FONTE_PEQUENA = ("Helvetica", 9)
FONTE_MONO = ("Courier New", 10)
FONTE_METRICA = ("Helvetica", 18, "bold")

# ── Dispositivo PyTorch ──────────────────────────────────────────────────────
DEVICE = (
    torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu")
    if TORCH_OK
    else None
)

print("Dispositivo:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =============================================================================
# WIDGETS REUTILIZÁVEIS
# =============================================================================


def rotulo_secao(pai, texto):
    ctk.CTkLabel(pai, text=texto, font=FONTE_SECAO, anchor="w").pack(
        anchor="w", padx=12, pady=(12, 4)
    )


def botao(pai, texto, comando, **kw):
    return ctk.CTkButton(pai, text=texto, font=FONTE_CORPO, command=comando, **kw)


class CartaoMetrica(ctk.CTkFrame):
    def __init__(self, pai, rotulo, valor="—"):
        super().__init__(pai, corner_radius=8, border_width=1)
        ctk.CTkLabel(self, text=rotulo, font=FONTE_PEQUENA).pack(pady=(8, 0))
        self._var = ctk.StringVar(value=valor)
        ctk.CTkLabel(self, textvariable=self._var, font=FONTE_METRICA).pack(pady=(0, 8))

    def definir(self, v):
        self._var.set(v)


class BarraStatus(ctk.CTkFrame):
    def __init__(self, pai):
        super().__init__(pai, corner_radius=0, height=26)
        self._var = ctk.StringVar(value="Pronto.")
        ctk.CTkLabel(self, textvariable=self._var, font=FONTE_PEQUENA, anchor="w").pack(
            side="left", padx=10
        )

    def definir(self, msg):
        self._var.set(msg)


# =============================================================================
# PIPELINE DE SEGMENTAÇÃO (independente da UI)
# =============================================================================


def organizar_registro(caminho_arquivo: str) -> dict | None:
    """
    Extrai metadados de uma imagem a partir do seu caminho no dataset LMLO.

    Regras:
    - Letra inicial define a classe:
        D → BI-RADS I  (classe 0)
        E → BI-RADS II (classe 1)
        F → BI-RADS III(classe 2)
        G → BI-RADS IV (classe 3)
    - Número múltiplo de 4 → teste; demais → treino
    """
    mapa = {"D": (0, "I"), "E": (1, "II"), "F": (2, "III"), "G": (3, "IV")}
    nome = os.path.basename(caminho_arquivo)
    letra = nome[0].upper() if nome else ""
    if letra not in mapa:
        return None
    classe, birads = mapa[letra]
    nome_sem_ext = os.path.splitext(nome)[0]
    digitos = re.sub(r"\D", "", nome_sem_ext)
    numero = int(digitos) if digitos else 0
    eh_treino = numero % 4 != 0
    return {
        "arquivo": caminho_arquivo,
        "letra": letra,
        "classe": classe,
        "birads": birads,
        "numero": numero,
        "treino": eh_treino,
    }


def limiar_otsu(arr_cinza: np.ndarray) -> int:
    """
    Calcula o limiar ótimo de Otsu para imagem uint8.

    Maximiza a variância inter-classes entre fundo (preto) e tecido mamário.
    Implementação manual baseada no artigo original de Otsu (1979).
    """
    histograma, _ = np.histogram(arr_cinza.flatten(), bins=256, range=(0, 256))
    total = arr_cinza.size
    soma_total = float(np.dot(np.arange(256), histograma))
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
        var = peso0 * peso1 * (media0 - media1) ** 2
        if var > melhor_var:
            melhor_var = var
            limiar = t
    return limiar


def maior_componente_conectado(mascara_bin: np.ndarray) -> np.ndarray:
    """
    Retorna máscara binária com apenas o maior componente conectado.

    Usado após Otsu para eliminar artefatos pequenos (anotações, réguas)
    que não fazem parte do tecido mamário principal.

    Usa scipy.ndimage com estrutura 3×3 (8-conectividade) quando disponível;
    caso contrário aplica erosão/dilatação via PIL como fallback.
    """
    if SCIPY_OK:
        estrutura = ndi.generate_binary_structure(2, 2)
        rotulado, n_comp = ndi.label(mascara_bin, structure=estrutura)
        if n_comp == 0:
            return mascara_bin.astype(np.uint8) * 255
        tamanhos = ndi.sum(mascara_bin, rotulado, range(1, n_comp + 1))
        maior = int(np.argmax(tamanhos)) + 1
        return (rotulado == maior).astype(np.uint8) * 255
    else:
        pil = Image.fromarray(mascara_bin.astype(np.uint8) * 255)
        pil = pil.filter(ImageFilter.MinFilter(9))
        pil = pil.filter(ImageFilter.MaxFilter(25))
        return np.array(pil)


def segmentar_mama(imagem_pil: Image.Image) -> Image.Image:
    """
    Pipeline robusto de segmentação da mama em imagem mamográfica.

    Etapas:
      1. Converte para escala de cinza normalizada uint8
      2. Limiarização de Otsu (fundo vs. tecido)
      3. Maior componente conectado (remove artefatos externos)
      4. Abertura morfológica: MinFilter(5) → MaxFilter(9)
         - MinFilter(5): erosão leve, remove ruídos de borda de 1-2px
         - MaxFilter(9): dilatação para recuperar a borda da mama
         - Tamanhos empíricos para imagens LMLO ~2000px de resolução
      5. Aplica máscara: mantém pixels da mama, zera fundo
    """
    arr = np.array(imagem_pil)
    if arr.dtype != np.uint8:
        mn, mx = arr.min(), arr.max()
        arr = ((arr.astype(np.float32) - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    limiar = limiar_otsu(arr)
    mascara = (arr > limiar).astype(np.uint8)
    mascara = maior_componente_conectado(mascara)
    pil_m = Image.fromarray(mascara)
    pil_m = pil_m.filter(ImageFilter.MinFilter(5))
    pil_m = pil_m.filter(ImageFilter.MaxFilter(9))
    arr_m = np.array(pil_m)
    resultado = np.where(arr_m > 0, arr, 0).astype(np.uint8)
    return Image.fromarray(resultado)


def recortar_bounding_box(imagem_seg: Image.Image) -> Image.Image:
    """
    Recorta a bounding box da mama segmentada, eliminando fundo desnecessário.

    Após segmentação, a imagem ainda contém grandes áreas pretas que degradam
    o treinamento (rede desperdiça capacidade modelando pixels nulos).
    O recorte garante que a mama ocupe ~100% da imagem resultante.

    Margem de 2px: evita cortar a borda exata do tecido.
    """
    arr = np.array(imagem_seg)
    linhas_validas = np.any(arr > 0, axis=1)
    cols_validas = np.any(arr > 0, axis=0)
    if not linhas_validas.any():
        return imagem_seg
    lin_min, lin_max = np.where(linhas_validas)[0][[0, -1]]
    col_min, col_max = np.where(cols_validas)[0][[0, -1]]
    lin_min = max(0, lin_min - 2)
    lin_max = min(arr.shape[0] - 1, lin_max + 2)
    col_min = max(0, col_min - 2)
    col_max = min(arr.shape[1] - 1, col_max + 2)
    return Image.fromarray(arr[lin_min : lin_max + 1, col_min : col_max + 1])


def preparar_imagem(imagem_pil: Image.Image) -> Image.Image:
    """
    Pipeline completo de preparação para DenseNet121/VGG16.

    Etapas:
      1. Segmentação robusta (Otsu + componente conectado + morfologia)
      2. Recorte do bounding box
      3. Redimensionamento para 224×224 (padrão ImageNet) com LANCZOS
         - LANCZOS oferece melhor qualidade no downscaling de alta resolução
      4. Conversão para RGB (replica canal cinza em R, G, B)
         - Necessário pois os pesos ImageNet esperam 3 canais
    """
    img_seg = segmentar_mama(imagem_pil)
    img_crop = recortar_bounding_box(img_seg)
    img_224 = img_crop.resize((224, 224), Image.LANCZOS)
    return img_224.convert("RGB")


def criar_dataloaders(
    dir_processado: str, batch_size: int = 32, num_workers: int = 0
) -> tuple:
    """
    Cria DataLoaders PyTorch a partir da estrutura processed/.

    Transformações:
    - ToTensor(): converte PIL [0,255] → tensor [0,1]
    - Normalize(mean, std): normalização ImageNet canônica
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      Usar esses valores é obrigatório pois as redes foram pré-treinadas com eles.

    Parâmetros:
    - batch_size=32: equilíbrio entre velocidade e estabilidade do gradiente
    - shuffle=True  no treino: previne overfitting por ordenação
    - shuffle=False no teste:  métricas reproduzíveis entre runs
    - num_workers=0: evita problemas de multiprocessing no Windows
    """
    if not TORCH_OK:
        return None, None
    media_imagenet = [0.485, 0.456, 0.406]
    desvio_imagenet = [0.229, 0.224, 0.225]
    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=media_imagenet, std=desvio_imagenet),
        ]
    )
    dir_treino = os.path.join(dir_processado, "train")
    dir_teste = os.path.join(dir_processado, "test")
    train_loader = test_loader = None
    if os.path.isdir(dir_treino):
        ds = datasets.ImageFolder(dir_treino, transform=transf)
        train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
    if os.path.isdir(dir_teste):
        ds = datasets.ImageFolder(dir_teste, transform=transf)
        test_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return train_loader, test_loader

def desativar_relu_inplace(modulo: nn.Module) -> nn.Module:
    """
    Percorre recursivamente todos os submódulos e converte ReLU(inplace=True)
    para ReLU(inplace=False).

    Por que é necessário:
    - VGG16 e DenseNet121 pré-treinados têm ReLU(inplace=True) em todo o backbone.
    - O Grad-CAM registra um hook no tensor de saída de uma camada intermediária
      via retain_grad() + register_hook().
    - Quando um ReLU subsequente opera inplace sobre esse tensor (ou um view dele),
      o PyTorch >= 2.0 detecta conflito com o BackwardHookFunctionBackward e lança
      RuntimeError.
    - Converter para inplace=False faz cada ReLU criar um novo tensor, eliminando
      o conflito sem alterar os valores computados.
    """
    for nome, submodulo in modulo.named_modules():
        if isinstance(submodulo, nn.ReLU) and submodulo.inplace:
            submodulo.inplace = False
    return modulo


# =============================================================================
# MODELO VGG16 — Transfer Learning
# =============================================================================


class ClassificadorVGG(nn.Module):
    def __init__(self, n_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        for param in base.features.parameters():
            param.requires_grad = False

        self.features = base.features
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(inplace=False),   # já corrigido na resposta anterior
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes),
        )

        # Converte TODOS os ReLU inplace do backbone (VGG tem ~13 deles)
        desativar_relu_inplace(self)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def descongelar_ultimas_conv(self, n_blocos: int = 1):
        pontos_bloco = [0, 5, 10, 17, 24]
        idx = pontos_bloco[max(0, len(pontos_bloco) - n_blocos - 1)]
        for i, camada in enumerate(self.features):
            if i >= idx:
                for param in camada.parameters():
                    param.requires_grad = True


# =============================================================================
# MODELO DENSENET121 — Transfer Learning
# =============================================================================


class ClassificadorDenseNet(nn.Module):
    def __init__(self, n_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        for param in base.features.parameters():
            param.requires_grad = False

        self.features = base.features
        num_features = base.classifier.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=False),   # já corrigido na resposta anterior
            nn.Dropout(p=dropout),
            nn.Linear(512, n_classes),
        )

        # DenseNet121 tem ReLU inplace espalhados por todos os dense blocks
        desativar_relu_inplace(self)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)   # torch.relu não é inplace por padrão
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def descongelar_ultimas_conv(self, n_blocos: int = 1):
        if n_blocos <= 0:
            return
        blocos = [
            ["denseblock4", "norm5"],
            ["denseblock3", "transition3", "denseblock4", "norm5"],
            ["denseblock2", "transition2", "denseblock3", "transition3", "denseblock4", "norm5"],
        ]
        nomes_para_descongelar = blocos[min(n_blocos, 3) - 1]
        for nome, modulo in self.features.named_children():
            if nome in nomes_para_descongelar:
                for param in modulo.parameters():
                    param.requires_grad = True


def camada_alvo_gradcam(modelo: nn.Module):
    """Escolhe automaticamente uma camada alvo adequada para Grad-CAM."""
    if isinstance(modelo, ClassificadorVGG):
        return modelo.features[28], "VGG16 features[28]"
    if isinstance(modelo, ClassificadorDenseNet):
        return modelo.features.norm5, "DenseNet121 features.norm5"
    if hasattr(modelo, "features"):
        return modelo.features[-1], "features[-1]"
    raise ValueError("Não foi possível definir a camada alvo do Grad-CAM para este modelo.")


# =============================================================================
# FUNÇÕES DE MÉTRICAS
# =============================================================================


def calcular_metricas_binario(y_true: list, y_pred: list) -> dict:
    """
    Calcula métricas para classificação binária.

    Positivo = classe 1 (III+IV — alta densidade, maior risco)
    Negativo = classe 0 (I+II — baixa densidade)

    Fórmulas:
    - Sensibilidade (Recall): TP / (TP + FN)
      Capacidade de detectar casos de alta densidade (crítica clinicamente)
    - Especificidade: TN / (TN + FP)
      Capacidade de confirmar casos de baixa densidade
    - Precisão: TP / (TP + FP)
    - Acurácia: (TP + TN) / Total
    - F1: 2 * Precisão * Sensibilidade / (Precisão + Sensibilidade)
      Harmônica entre precisão e sensibilidade; robusto a classes desbalanceadas
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    espec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    return {
        "sensibilidade": sens,
        "especificidade": espec,
        "precisao": prec,
        "acuracia": acc,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calcular_metricas_4classes(y_true: list, y_pred: list, n_classes: int = 4) -> dict:
    """
    Calcula sensibilidade e especificidade médias para 4 classes (One-vs-Rest).

    Para cada classe c:
    - Sensibilidade_c = TP_c / (TP_c + FN_c)
    - Especificidade_c = TN_c / (TN_c + FP_c)

    Média macro: trata todas as classes igualmente, independente do suporte.
    Preferida quando as classes têm importância clínica similar.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matriz[t][p] += 1
    sens_por_classe = []
    espec_por_classe = []
    for c in range(n_classes):
        tp = matriz[c, c]
        fn = matriz[c, :].sum() - tp
        fp = matriz[:, c].sum() - tp
        tn = matriz.sum() - tp - fn - fp
        sens_por_classe.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        espec_por_classe.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return {
        "sensibilidade_media": float(np.mean(sens_por_classe)),
        "especificidade_media": float(np.mean(espec_por_classe)),
        "sensibilidade_por_classe": sens_por_classe,
        "especificidade_por_classe": espec_por_classe,
        "matriz_confusao": matriz.tolist(),
    }


# =============================================================================
# LOOP DE TREINAMENTO REAL
# =============================================================================


def treinar_modelo(
    modelo,
    train_loader,
    test_loader,
    n_epocas: int,
    lr: float,
    weight_decay: float,
    modo: str,
    callback_epoca=None,
    callback_fim=None,
    parar_flag=None,
) -> dict:
    """
    Loop de treinamento completo com coleta de histórico por época.

    Estratégia de otimização:
    - Otimizador Adam: lr adaptativo por parâmetro; robusto a gradientes
      esparsas; poucas épocas de warm-up necessárias vs SGD.
      weight_decay implementa L2 regularization diretamente.
    - Scheduler ReduceLROnPlateau: reduz lr por fator 0.5 quando a val_loss
      para de melhorar por 3 épocas consecutivas (patience=3).
      Mais adaptativo que StepLR; responde ao comportamento real da perda.
    - Early stopping: interrompe quando val_loss não melhora por 5 épocas.
      Salva o melhor modelo por val_loss, não pelo último checkpoint.
      Previne overfitting sem fixar um número de épocas arbitrário.

    Parâmetros:
    - modo: "binario" (2 saídas) ou "quadriclasse" (4 saídas)
    - callback_epoca(dict): chamado ao fim de cada época para atualizar UI
    - callback_fim(dict):   chamado ao fim do treinamento com histórico total
    - parar_flag: threading.Event; se set(), interrompe o treinamento

    Retorna dicionário com histórico completo de todas as métricas por época.
    """
    if not TORCH_OK:
        return {}

    modelo = modelo.to(DEVICE)
    criterio = nn.CrossEntropyLoss()

    # Apenas parâmetros treináveis (classifier + eventuais conv descongeladas)
    params_treinaveis = filter(lambda p: p.requires_grad, modelo.parameters())
    otimizador = optim.Adam(params_treinaveis, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        otimizador, mode="min", factor=0.5, patience=3
    )

    n_treino = len(train_loader.dataset) if train_loader else 0
    n_teste = len(test_loader.dataset) if test_loader else 0
    _logger.info(
        f"Início do treinamento | modo={modo} | épocas={n_epocas} | "
        f"lr={lr} | wd={weight_decay} | train={n_treino} | test={n_teste}"
    )

    historico = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "sensibilidade": [],
        "especificidade": [],
        "precisao": [],
        "acuracia": [],
        "f1": [],
        "lr_por_epoca": [],
        "tempo_por_epoca": [],
        "epocas_rodadas": 0,
    }

    melhor_val_loss = float("inf")
    melhor_estado = None
    paciencia_atual = 0
    paciencia_max = 5  # early stopping após 5 épocas sem melhora

    for epoca in range(1, n_epocas + 1):
        if parar_flag and parar_flag.is_set():
            break

        t0 = time.time()

        # ── FASE DE TREINO ──────────────────────────────────────────────────
        modelo.train()
        total_loss_tr = 0.0
        acertos_tr = 0
        total_tr = 0

        for imgs, labels in train_loader or []:
            if parar_flag and parar_flag.is_set():
                break

            # Mapeia labels para binário se necessário
            if modo == "binario":
                labels = (labels >= 2).long()

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            otimizador.zero_grad()
            saidas = modelo(imgs)
            loss = criterio(saidas, labels)
            loss.backward()
            otimizador.step()

            total_loss_tr += loss.item() * imgs.size(0)
            preds = saidas.argmax(dim=1)
            acertos_tr += (preds == labels).sum().item()
            total_tr += imgs.size(0)

        loss_tr = total_loss_tr / max(total_tr, 1)
        acc_tr = acertos_tr / max(total_tr, 1)

        # ── FASE DE VALIDAÇÃO ───────────────────────────────────────────────
        modelo.eval()
        total_loss_val = 0.0
        acertos_val = 0
        total_val = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for imgs, labels in test_loader or []:
                if modo == "binario":
                    labels = (labels >= 2).long()
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                saidas = modelo(imgs)
                loss = criterio(saidas, labels)
                total_loss_val += loss.item() * imgs.size(0)
                preds = saidas.argmax(dim=1)
                acertos_val += (preds == labels).sum().item()
                total_val += imgs.size(0)
                y_true_val.extend(labels.cpu().numpy().tolist())
                y_pred_val.extend(preds.cpu().numpy().tolist())

        loss_val = total_loss_val / max(total_val, 1)
        acc_val = acertos_val / max(total_val, 1)

        # Métricas detalhadas de validação
        if modo == "binario":
            met = calcular_metricas_binario(y_true_val, y_pred_val)
        else:
            met4 = calcular_metricas_4classes(y_true_val, y_pred_val)
            # Para o gráfico, usa sensib/espec médias como proxy
            met = {
                "sensibilidade": met4["sensibilidade_media"],
                "especificidade": met4["especificidade_media"],
                "precisao": acc_val,  # placeholder
                "acuracia": acc_val,
                "f1": (met4["sensibilidade_media"] + met4["especificidade_media"]) / 2,
            }

        scheduler.step(loss_val)
        lr_atual = otimizador.param_groups[0]["lr"]
        tempo_ep = time.time() - t0

        # Registra no histórico
        historico["train_loss"].append(round(loss_tr, 4))
        historico["val_loss"].append(round(loss_val, 4))
        historico["train_acc"].append(round(acc_tr, 4))
        historico["val_acc"].append(round(acc_val, 4))
        historico["sensibilidade"].append(round(met["sensibilidade"], 4))
        historico["especificidade"].append(round(met["especificidade"], 4))
        historico["precisao"].append(round(met["precisao"], 4))
        historico["acuracia"].append(round(met["acuracia"], 4))
        historico["f1"].append(round(met["f1"], 4))
        historico["lr_por_epoca"].append(lr_atual)
        historico["tempo_por_epoca"].append(round(tempo_ep, 2))
        historico["epocas_rodadas"] = epoca

        # Early stopping
        if loss_val < melhor_val_loss - 1e-4:
            melhor_val_loss = loss_val
            melhor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            paciencia_atual = 0
        else:
            paciencia_atual += 1
            if paciencia_atual >= paciencia_max:
                _logger.info(
                    f"Early stopping acionado na época {epoca} "
                    f"(sem melhora por {paciencia_max} épocas)"
                )
                if callback_epoca:
                    callback_epoca(
                        {**historico, "epoca": epoca, "status": "early_stop"}
                    )
                break

        _logger.info(
            f"Época {epoca}/{n_epocas} | "
            f"Loss={loss_tr:.4f}/{loss_val:.4f} | "
            f"Acc={acc_tr:.3f}/{acc_val:.3f} | "
            f"LR={lr_atual:.2e} | {tempo_ep:.1f}s"
        )

        if callback_epoca:
            callback_epoca(
                {**historico, "epoca": epoca, "lr_atual": lr_atual, "status": "ok"}
            )

    # Restaura melhor modelo
    if melhor_estado:
        modelo.load_state_dict(melhor_estado)

    historico["melhor_val_loss"] = round(melhor_val_loss, 4)

    # Calcula melhor acurácia e melhor F1 para o resumo final
    melhor_acc = max(historico["val_acc"]) if historico["val_acc"] else 0.0
    melhor_f1 = max(historico["f1"]) if historico["f1"] else 0.0
    tempo_total = sum(historico["tempo_por_epoca"])
    historico["melhor_acc"] = round(melhor_acc, 4)
    historico["melhor_f1"] = round(melhor_f1, 4)
    historico["tempo_total"] = round(tempo_total, 2)

    _logger.info(
        f"Treinamento concluído | épocas={historico['epocas_rodadas']} | "
        f"melhor val_acc={melhor_acc:.3f} | melhor F1={melhor_f1:.3f} | "
        f"tempo total={tempo_total:.1f}s"
    )

    if callback_fim:
        callback_fim(historico)
    return historico


# =============================================================================
# GRAD-CAM REAL
# =============================================================================


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping para VGG16.

    Referência: Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization", ICCV 2017.

    Funcionamento:
    1. Registra hooks na última camada convolucional (features[28] no VGG16)
    2. Forward pass captura os feature maps A^k (ativações)
    3. Backward pass captura os gradientes ∂y^c/∂A^k
    4. Pondera cada feature map pelo gradiente médio global (GAP dos gradientes)
    5. Aplica ReLU: mantém apenas contribuições positivas para a classe
    6. Normaliza e redimensiona para 224×224

    Por que a última camada conv?
    - Mantém a resolução espacial mais alta antes do pooling final
    - Captura features de alto nível (texturas densas específicas de BIRADS)
    - Camadas mais cedo têm features muito genéricas (bordas, cores)
    """

    def __init__(self, modelo: nn.Module, camada_alvo=None):
        self.modelo = modelo
        self._ativacoes = None
        self._gradientes = None
        self._hooks = []

        # Escolhe camada alvo automaticamente:
        # VGG16 → features[28]
        # DenseNet121 → features.norm5
        if camada_alvo is None:
            camada_alvo, self.nome_camada_alvo = camada_alvo_gradcam(modelo)
        else:
            self.nome_camada_alvo = str(camada_alvo)

        # Hook forward: captura ativações A^k após a camada alvo
        self._hooks.append(camada_alvo.register_forward_hook(self._salvar_ativacoes))
        # Hook backward: captura gradientes ∂y^c/∂A^k
        self._hooks.append(
            camada_alvo.register_full_backward_hook(self._salvar_gradientes)
        )

    def _salvar_ativacoes(self, modulo, entrada, saida):
        self._ativacoes = saida.detach()

    def _salvar_gradientes(self, modulo, grad_entrada, grad_saida):
        self._gradientes = grad_saida[0].detach()

    def gerar(self, tensor_img: "torch.Tensor", classe_idx: int = None) -> np.ndarray:
        """
        Gera o mapa de calor Grad-CAM para a classe especificada.

        Parâmetros:
        - tensor_img: tensor [1, 3, 224, 224] normalizado (ImageNet)
        - classe_idx: índice da classe alvo; None → usa a classe predita

        Retorna:
        - Array float32 [224, 224] normalizado [0, 1] representando
          a relevância de cada pixel para a decisão da rede.
        """
        self.modelo.eval()
        tensor_img = tensor_img.to(DEVICE).requires_grad_(True)
        saida = self.modelo(tensor_img)

        if classe_idx is None:
            classe_idx = saida.argmax(dim=1).item()

        self.modelo.zero_grad()
        # Gradiente apenas para a classe alvo
        saida[0, classe_idx].backward()

        # α^c_k = (1/Z) Σ_ij (∂y^c / ∂A^k_ij)
        pesos = self._gradientes.mean(dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]

        # L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)
        mapa = (pesos * self._ativacoes).sum(dim=1, keepdim=True)  # [1,1,7,7]
        mapa = torch.relu(mapa)

        # Normaliza e redimensiona para 224×224
        mapa_np = mapa.squeeze().cpu().numpy()
        if mapa_np.max() > 0:
            mapa_np = mapa_np / mapa_np.max()

        # Upsampling bilinear via PIL
        mapa_pil = Image.fromarray((mapa_np * 255).astype(np.uint8))
        mapa_pil = mapa_pil.resize((224, 224), Image.BILINEAR)
        return np.array(mapa_pil) / 255.0

    def remover_hooks(self):
        for h in self._hooks:
            h.remove()


def aplicar_colormap_jet(mapa: np.ndarray) -> np.ndarray:
    """
    Aplica colormap Jet vetorizado ao mapa de calor Grad-CAM.

    Jet: azul (baixa ativação) → verde → amarelo → vermelho (alta ativação)
    Implementação sem matplotlib: permite uso em qualquer contexto.

    Fórmula vetorizada derivada do colormap Jet do matplotlib.
    """
    r = np.clip(1.5 - np.abs(4 * mapa - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * mapa - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * mapa - 1), 0, 1)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


# =============================================================================
# GRÁFICOS DE CONVERGÊNCIA
# =============================================================================


def gerar_graficos_convergencia(historico: dict, titulo: str = "VGG16") -> Image.Image:
    """
    Gera painel de gráficos de convergência a partir do histórico de treinamento.

    Layout: grade 2×3 com:
    1. Loss (treino vs validação)
    2. Acurácia (treino vs validação)
    3. Sensibilidade e Especificidade (validação)
    4. Precisão e F1 (validação)
    5. Learning Rate por época (escala log)
    6. Tempo por época

    Retorna imagem PIL para exibição na interface sem salvar em disco.

    Por que esses gráficos?
    - Loss curves: diagnóstico principal de overfitting (gap treino-val)
    - Accuracy: intuição imediata da performance
    - Sensibilidade/Especificidade: métricas clínicas mais relevantes
    - LR: visualização do efeito do scheduler ReduceLROnPlateau
    - Tempo: importante para discutir viabilidade prática
    """
    if not MPL_OK or not historico.get("train_loss"):
        return None

    epocas = list(range(1, len(historico["train_loss"]) + 1))

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Convergência do Treinamento — {titulo}", fontsize=14, fontweight="bold"
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    cores = {
        "treino": "#2176AE",
        "val": "#E83151",
        "sens": "#0D7377",
        "espec": "#F4A261",
        "prec": "#8338EC",
        "f1": "#3A86FF",
        "lr": "#6D6875",
        "tempo": "#457B9D",
    }

    def _ax(linha, col):
        return fig.add_subplot(gs[linha, col])

    # ── 1. Loss ──────────────────────────────────────────────────────────────
    ax = _ax(0, 0)
    ax.plot(
        epocas,
        historico["train_loss"],
        color=cores["treino"],
        lw=2,
        label="Treino",
        marker="o",
        ms=3,
    )
    ax.plot(
        epocas,
        historico["val_loss"],
        color=cores["val"],
        lw=2,
        label="Validação",
        marker="o",
        ms=3,
        linestyle="--",
    )
    ax.set_title("Loss (Cross-Entropy)", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 2. Acurácia ───────────────────────────────────────────────────────────
    ax = _ax(0, 1)
    ax.plot(
        epocas,
        [v * 100 for v in historico["train_acc"]],
        color=cores["treino"],
        lw=2,
        label="Treino",
        marker="o",
        ms=3,
    )
    ax.plot(
        epocas,
        [v * 100 for v in historico["val_acc"]],
        color=cores["val"],
        lw=2,
        label="Validação",
        marker="o",
        ms=3,
        linestyle="--",
    )
    ax.set_title("Acurácia", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 3. Sensibilidade e Especificidade ────────────────────────────────────
    ax = _ax(0, 2)
    ax.plot(
        epocas,
        [v * 100 for v in historico["sensibilidade"]],
        color=cores["sens"],
        lw=2,
        label="Sensibilidade",
        marker="s",
        ms=3,
    )
    ax.plot(
        epocas,
        [v * 100 for v in historico["especificidade"]],
        color=cores["espec"],
        lw=2,
        label="Especificidade",
        marker="^",
        ms=3,
        linestyle="--",
    )
    ax.set_title("Sensibilidade & Especificidade", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Métrica (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 4. Precisão e F1 ─────────────────────────────────────────────────────
    ax = _ax(1, 0)
    ax.plot(
        epocas,
        [v * 100 for v in historico["precisao"]],
        color=cores["prec"],
        lw=2,
        label="Precisão",
        marker="D",
        ms=3,
    )
    ax.plot(
        epocas,
        [v * 100 for v in historico["f1"]],
        color=cores["f1"],
        lw=2,
        label="F1",
        marker="o",
        ms=3,
        linestyle="--",
    )
    ax.set_title("Precisão & F1", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Métrica (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 5. Learning Rate ─────────────────────────────────────────────────────
    ax = _ax(1, 1)
    ax.plot(
        epocas, historico["lr_por_epoca"], color=cores["lr"], lw=2, marker="o", ms=3
    )
    ax.set_title("Learning Rate (ReduceLROnPlateau)", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # ── 6. Tempo por época ───────────────────────────────────────────────────
    ax = _ax(1, 2)
    ax.bar(
        epocas, historico["tempo_por_epoca"], color=cores["tempo"], alpha=0.8, width=0.6
    )
    ax.set_title("Tempo por Época", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Tempo (s)")
    ax.grid(alpha=0.3, axis="y")

    # Renderiza para buffer PIL (sem salvar em disco)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =============================================================================
# ABA VISUALIZADOR
# =============================================================================


class AbaVisualizador(ctk.CTkFrame):
    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status = status
        self._img_original = None
        self._img_segmentada = None
        self._zoom = 1.0
        self._mostrar_mascara = False
        self._construir()

    def _construir(self):
        painel = ctk.CTkFrame(self, width=220, corner_radius=10)
        painel.pack(side="left", fill="y", padx=(0, 6))
        painel.pack_propagate(False)

        rotulo_secao(painel, "IMAGEM")
        botao(painel, "📂 Abrir PNG/TIFF", self._abrir).pack(padx=12, fill="x")

        rotulo_secao(painel, "ZOOM")
        self._lbl_zoom = ctk.CTkLabel(painel, text="100%", font=FONTE_CORPO)
        self._lbl_zoom.pack()
        self._slider_zoom = ctk.CTkSlider(
            painel, from_=0.2, to=4.0, number_of_steps=38, command=self._ao_zoom
        )
        self._slider_zoom.set(1.0)
        self._slider_zoom.pack(padx=12, fill="x")
        botao(
            painel,
            "Reset 1:1",
            self._reset_zoom,
            border_width=1,
        ).pack(padx=12, pady=4, fill="x")

        rotulo_secao(painel, "SEGMENTAÇÃO")
        botao(painel, "⚙ Segmentar Mama", self._segmentar).pack(padx=12, fill="x")
        self._btn_mascara = botao(
            painel,
            "👁 Ver Máscara",
            self._alternar_mascara,
            border_width=1,
            state="disabled",
        )
        self._btn_mascara.pack(padx=12, pady=4, fill="x")

        rotulo_secao(painel, "INFO")
        self._caixa_info = ctk.CTkTextbox(
            painel, height=130, font=FONTE_MONO, state="disabled"
        )
        self._caixa_info.pack(padx=12, fill="x")

        area = ctk.CTkFrame(self, corner_radius=10)
        area.pack(side="left", fill="both", expand=True)
        self._titulo_img = ctk.CTkLabel(
            area, text="Nenhuma imagem carregada", font=FONTE_SECAO, anchor="w"
        )
        self._titulo_img.pack(anchor="w", padx=12, pady=(8, 4))
        fundo_canvas = ctk.CTkFrame(area, corner_radius=6)
        fundo_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._canvas = tk.Canvas(fundo_canvas, highlightthickness=0, bg="#f0f0f0")
        sv = ctk.CTkScrollbar(fundo_canvas, command=self._canvas.yview)
        sh = ctk.CTkScrollbar(
            fundo_canvas, orientation="horizontal", command=self._canvas.xview
        )
        self._canvas.configure(yscrollcommand=sv.set, xscrollcommand=sh.set)
        sh.pack(side="bottom", fill="x")
        sv.pack(side="right", fill="y")
        self._canvas.pack(fill="both", expand=True)

    def _abrir(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        try:
            self._img_original = Image.open(caminho)
            self._img_segmentada = None
            self._mostrar_mascara = False
            self._btn_mascara.configure(state="disabled")
            self._slider_zoom.set(1.0)
            self._zoom = 1.0
            self._titulo_img.configure(text=os.path.basename(caminho))
            self._atualizar_info(caminho)
            self._renderizar()
            self._status.definir(f"Imagem: {os.path.basename(caminho)}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _atualizar_info(self, caminho):
        img = self._img_original
        texto = (
            f"Arquivo: {os.path.basename(caminho)}\n"
            f"Tamanho: {img.width}×{img.height} px\n"
            f"Modo:    {img.mode}\n"
            f"Disco:   {os.path.getsize(caminho) / 1024:.1f} KB"
        )
        self._caixa_info.configure(state="normal")
        self._caixa_info.delete("1.0", "end")
        self._caixa_info.insert("1.0", texto)
        self._caixa_info.configure(state="disabled")

    def _ao_zoom(self, val):
        self._zoom = float(val)
        self._lbl_zoom.configure(text=f"{int(self._zoom * 100)}%")
        self._renderizar()

    def _reset_zoom(self):
        self._slider_zoom.set(1.0)
        self._ao_zoom(1.0)

    def _renderizar(self):
        if not self._img_original:
            return
        fonte = (
            self._img_segmentada
            if self._mostrar_mascara and self._img_segmentada
            else self._img_original
        )
        arr = np.array(fonte)
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(
                np.uint8
            )
        pil = Image.fromarray(arr)
        if pil.mode not in ("L", "RGB", "RGBA"):
            pil = pil.convert("L")
        larg = max(1, int(pil.width * self._zoom))
        alt = max(1, int(pil.height * self._zoom))
        pil = pil.resize((larg, alt), Image.LANCZOS)
        self._img_tk = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._img_tk)
        self._canvas.configure(scrollregion=(0, 0, larg, alt))

    def _segmentar(self):
        if not self._img_original:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        self._status.definir("Segmentando…")
        threading.Thread(target=self._executar_segmentacao, daemon=True).start()

    def _executar_segmentacao(self):
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
            text="👁 Ver Original" if self._mostrar_mascara else "👁 Ver Máscara"
        )
        self._renderizar()


# =============================================================================
# ABA DATASET
# =============================================================================


class AbaDataset(ctk.CTkFrame):
    """
    Aba de preparação dos dados:
    - Leitura e organização automática do dataset LMLO
    - Segmentação, recorte e redimensionamento (224×224 RGB)
    - Data Augmentation por rotação (treino apenas)
    - Geração da estrutura processed/ para uso pelas redes
    - Criação de DataLoaders PyTorch
    """

    MAPA_CLASSE = {"D": (0, "I"), "E": (1, "II"), "F": (2, "III"), "G": (3, "IV")}
    ANGULOS_AUG = [-20, -10, 0, 10, 20]
    TAMANHO_ALVO = (224, 224)

    def __init__(self, pai, status):
        super().__init__(pai, fg_color="transparent")
        self._status = status
        self._registros: list[dict] = []
        self._imgs_treino: list[str] = []
        self._imgs_teste: list[str] = []
        self._dir_dataset = ""
        self._dir_processado = ""
        self._train_loader = None
        self._test_loader = None
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

        frame_classes = ctk.CTkFrame(self, corner_radius=10)
        frame_classes.pack(fill="x", pady=(0, 6))
        rotulo_secao(frame_classes, "CLASSES BIRADS")
        linha_cards = ctk.CTkFrame(frame_classes, fg_color="transparent")
        linha_cards.pack(fill="x", padx=12, pady=(0, 10))
        self._cards_classe = [
            CartaoMetrica(linha_cards, f"BIRADS {r}") for r in ("I", "II", "III", "IV")
        ]
        [
            c.pack(side="left", expand=True, fill="both", padx=3)
            for c in self._cards_classe
        ]

        linha_split = ctk.CTkFrame(frame_classes, fg_color="transparent")
        linha_split.pack(fill="x", padx=12, pady=(0, 10))
        self._card_treino = CartaoMetrica(linha_split, "Treino")
        self._card_teste = CartaoMetrica(linha_split, "Teste (múlt. 4)")
        self._card_total = CartaoMetrica(linha_split, "Total")
        for c in (self._card_treino, self._card_teste, self._card_total):
            c.pack(side="left", expand=True, fill="both", padx=3)

        frame_aumento = ctk.CTkFrame(self, corner_radius=10)
        frame_aumento.pack(fill="x", pady=(0, 6))
        rotulo_secao(frame_aumento, "AUMENTO DE DADOS")
        ctk.CTkLabel(
            frame_aumento,
            font=FONTE_CORPO,
            text="Rotações: −20° −10° 0° +10° +20°  (5× por imagem)",
        ).pack(anchor="w", padx=12)
        linha_aug = ctk.CTkFrame(frame_aumento, fg_color="transparent")
        linha_aug.pack(fill="x", padx=12, pady=(4, 10))
        botao(linha_aug, "⟳ Realizar Aumento", self.realizarAugmentacao).pack(
            side="left"
        )
        self._barra_aumento = ctk.CTkProgressBar(linha_aug)
        self._barra_aumento.set(0)
        self._barra_aumento.pack(side="left", fill="x", expand=True, padx=8)
        self._lbl_aumento = ctk.CTkLabel(
            linha_aug, text="0/0", font=FONTE_PEQUENA, width=50
        )
        self._lbl_aumento.pack(side="left")

        frame_log = ctk.CTkFrame(self, corner_radius=10)
        frame_log.pack(fill="both", expand=True)
        rotulo_secao(frame_log, "LOG")
        self._log = ctk.CTkTextbox(frame_log, font=FONTE_MONO, state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def _registrar(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_ts(self, msg):
        self.after(0, lambda m=msg: self._registrar(m))

    def _carregar_dir(self):
        diretorio = filedialog.askdirectory()
        if not diretorio:
            return
        self._dir_dataset = diretorio
        self._lbl_dir.configure(text=diretorio)
        extensoes_validas = {".png", ".tif", ".tiff"}
        todos_caminhos = []
        for raiz, _, arquivos in os.walk(diretorio):
            for arq in sorted(arquivos):
                if os.path.splitext(arq)[1].lower() in extensoes_validas:
                    todos_caminhos.append(os.path.join(raiz, arq))
        if not todos_caminhos:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada.")
            return
        self._registros = []
        contagem_classe = [0, 0, 0, 0]
        treino, teste = [], []
        for caminho in todos_caminhos:
            rec = organizar_registro(caminho)
            if rec is None:
                continue
            self._registros.append(rec)
            contagem_classe[rec["classe"]] += 1
            if rec["treino"]:
                treino.append(caminho)
            else:
                teste.append(caminho)
        self._imgs_treino = treino
        self._imgs_teste = teste
        for i in range(4):
            self._cards_classe[i].definir(str(contagem_classe[i]))
        self._card_treino.definir(str(len(treino)))
        self._card_teste.definir(str(len(teste)))
        self._card_total.definir(str(len(todos_caminhos)))
        self._registrar(log_info(f"Dataset carregado: {diretorio}"))
        self._registrar(
            log_info(
                f"Total: {len(todos_caminhos)} imagens | "
                f"Treino: {len(treino)} | Teste: {len(teste)}"
            )
        )
        for letra, (idx, birads) in self.MAPA_CLASSE.items():
            self._registrar(
                f"  BI-RADS {birads} ({letra}): {contagem_classe[idx]} imagens"
            )
        self._status.definir(f"Dataset: {len(todos_caminhos)} imagens carregadas.")
        threading.Thread(target=self._executar_processamento, daemon=True).start()

    def _executar_processamento(self):
        total = len(self._registros)
        if total == 0:
            return
        self._dir_processado = os.path.join(self._dir_dataset, "processed")
        for split in ("train", "test"):
            for letra in self.MAPA_CLASSE:
                os.makedirs(
                    os.path.join(self._dir_processado, split, letra), exist_ok=True
                )
        self._log_ts(log_info("Iniciando processamento (segmentação + crop + resize)…"))
        self.after(0, lambda: self._status.definir("Processando imagens…"))
        for i, rec in enumerate(self._registros):
            caminho = rec["arquivo"]
            letra = rec["letra"]
            split = "train" if rec["treino"] else "test"
            nome_arq = os.path.basename(caminho)
            destino = os.path.join(self._dir_processado, split, letra, nome_arq)
            try:
                img = Image.open(caminho)
                img_proc = preparar_imagem(img)
                img_proc.save(destino)
            except Exception as e:
                self._log_ts(log_erro(f"{nome_arq}: {e}"))
            if (i + 1) % 10 == 0 or (i + 1) == total:
                self._log_ts(log_info(f"Segmentação: {i + 1}/{total} imagens processadas"))
                self.after(0, lambda v=(i + 1) / total: self._barra_aumento.set(v))
        if TORCH_OK:
            self._train_loader, self._test_loader = criar_dataloaders(
                self._dir_processado
            )
            n_tr = len(self._train_loader.dataset) if self._train_loader else 0
            n_te = len(self._test_loader.dataset) if self._test_loader else 0
            self._log_ts(log_info(f"DataLoaders criados: train={n_tr}, test={n_te}"))
        else:
            self._log_ts(log_warn("PyTorch não encontrado — DataLoaders não criados."))
        self.after(
            0,
            lambda: (
                self._barra_aumento.set(1.0),
                self._status.definir("Processamento concluído."),
            ),
        )
        self._log_ts(log_info(f"Estrutura processada salva em: {self._dir_processado}"))

    def realizarAugmentacao(self):
        if not self._imgs_treino:
            messagebox.showwarning("Aviso", "Carregue um dataset primeiro.")
            return
        if not self._dir_processado or not os.path.isdir(self._dir_processado):
            messagebox.showwarning("Aviso", "Aguarde o processamento ser concluído.")
            return
        threading.Thread(target=self._executar_aumento, daemon=True).start()

    def _executar_aumento(self):
        """
        Data Augmentation real — apenas treino.

        Rotações: -20, -10, 0, +10, +20 graus
        - Intervalo de ±20° empiricamente validado para mamografias:
          pequeno o suficiente para não distorcer a anatomia,
          grande o suficiente para regularizar a orientação do exame.
        - expand=False: mantém 224×224 após a rotação
        - fillcolor=(0,0,0): preenche bordas com preto (fundo padrão)
        - resample=BILINEAR: interpolação suave, evita artefatos de aliasing

        Nomenclatura dos arquivos: _rot_m20 (minus 20), _rot_p20 (plus 20), _rot_0.
        """
        sufixos = {-20: "m20", -10: "m10", 0: "0", 10: "p10", 20: "p20"}
        registros_treino = [r for r in self._registros if r["treino"]]
        total = len(registros_treino) * len(self.ANGULOS_AUG)
        feito = 0
        geradas = 0

        # Contagem original antes do aumento
        n_treino_original = len(registros_treino)

        self._log_ts(log_info(
            f"Aumento iniciado: {n_treino_original} imgs × "
            f"{len(self.ANGULOS_AUG)} rotações = {total} arquivos esperados"
        ))
        self._log_ts(
            f"  Dataset original — Treino: {n_treino_original} | Teste: {len(self._imgs_teste)}"
        )
        self.after(0, lambda: self._barra_aumento.set(0))
        self.after(0, lambda: self._status.definir("Realizando aumento…"))
        for rec in registros_treino:
            letra = rec["letra"]
            nome_arq = os.path.basename(rec["arquivo"])
            nome_sem_ext = os.path.splitext(nome_arq)[0]
            caminho_proc = os.path.join(self._dir_processado, "train", letra, nome_arq)
            try:
                img_proc = (
                    Image.open(caminho_proc)
                    if os.path.isfile(caminho_proc)
                    else preparar_imagem(Image.open(rec["arquivo"]))
                )
            except Exception as e:
                self._log_ts(log_erro(f"{nome_arq}: {e}"))
                feito += len(self.ANGULOS_AUG)
                continue
            for ang in self.ANGULOS_AUG:
                suf = sufixos[ang]
                nome_aug = f"{nome_sem_ext}_rot_{suf}.png"
                destino = os.path.join(self._dir_processado, "train", letra, nome_aug)
                img_rot = img_proc.rotate(
                    ang, expand=False, fillcolor=(0, 0, 0), resample=Image.BILINEAR
                )
                img_rot.save(destino)
                geradas += 1
                feito += 1
                fracao = feito / total
                self.after(
                    0,
                    lambda f=fracao, d=feito, t=total: (
                        self._barra_aumento.set(f),
                        self._lbl_aumento.configure(text=f"{d}/{t}"),
                    ),
                )

        # Recria dataloaders e obtém contagem final
        n_treino_apos = n_treino_original  # fallback
        if TORCH_OK and self._dir_processado:
            self._train_loader, self._test_loader = criar_dataloaders(
                self._dir_processado
            )
            n_treino_apos = len(self._train_loader.dataset) if self._train_loader else geradas
            self._log_ts(log_info(f"DataLoaders recriados: train={n_treino_apos} (com augmentation)"))

        # ITEM 1: exibir resumo claro antes/depois na interface e no terminal
        resumo = (
            f"\n{'='*45}\n"
            f"  RESUMO DO DATASET APÓS AUGMENTATION\n"
            f"{'='*45}\n"
            f"  Dataset original:\n"
            f"    Treino : {n_treino_original} imagens\n"
            f"    Teste  : {len(self._imgs_teste)} imagens\n"
            f"\n"
            f"  Após augmentation:\n"
            f"    Treino : {n_treino_apos} imagens\n"
            f"    Teste  : {len(self._imgs_teste)} imagens (inalterado)\n"
            f"    Geradas: {geradas} novas imagens\n"
            f"{'='*45}"
        )
        self._log_ts(log_info(f"Augmentation concluída: {geradas} imagens geradas"))
        _logger.info(resumo)  # imprime também no terminal

        self.after(
            0,
            lambda: (
                self._registrar(resumo),
                self._card_treino.definir(f"{n_treino_apos}↑"),
                self._status.definir(
                    f"Augmentation concluída — treino: {n_treino_original}→{n_treino_apos}"
                ),
            ),
        )

    # ── Propriedades acessíveis pelas outras abas ────────────────────────────
    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def dir_processado(self):
        return self._dir_processado


# =============================================================================
# ABA CLASSIFICAÇÃO — VGG16 REAL
# =============================================================================


class AbaClassificacao(ctk.CTkFrame):
    """
    Aba de treinamento e classificação com VGG16 real.

    Fluxo:
    1. Configura hiperparâmetros (lr, dropout, épocas, modo)
    2. Cria ClassificadorVGG e inicia treinar_modelo() em thread separada
    3. Cada época chama _atualizar_ui_epoca() via self.after() (thread-safe)
    4. Ao fim do treino, exibe gráficos de convergência no subpainel
    5. Botão "Classificar" avalia o modelo no conjunto de teste
    6. Botão "Salvar" persiste os pesos em .pth e o histórico em .json
    """

    def __init__(self, pai, status, aba_dataset: AbaDataset):
        super().__init__(pai, fg_color="transparent")
        self._status = status
        self._aba_dataset = aba_dataset
        self._modo = ctk.StringVar(value="binario")
        self._modelo_nome = ctk.StringVar(value="vgg")
        self._modelo = None
        self._historico = {}
        self._parar_flag = threading.Event()
        self._img_grafico_tk = None
        self._construir()

    def _construir(self):
        # ── Painel esquerdo: controles ───────────────────────────────────────
        painel_esq = ctk.CTkFrame(self, width=280, corner_radius=10)
        painel_esq.pack(side="left", fill="y", padx=(0, 6))
        painel_esq.pack_propagate(False)

        rotulo_secao(painel_esq, "MODO")
        ctk.CTkRadioButton(
            painel_esq,
            text="Binário (I+II vs III+IV)",
            variable=self._modo,
            value="binario",
            font=FONTE_CORPO,
        ).pack(anchor="w", padx=14, pady=2)
        ctk.CTkRadioButton(
            painel_esq,
            text="4 Classes (I×II×III×IV)",
            variable=self._modo,
            value="quadriclasse",
            font=FONTE_CORPO,
        ).pack(anchor="w", padx=14, pady=2)

        rotulo_secao(painel_esq, "MODELO")
        ctk.CTkRadioButton(
            painel_esq,
            text="VGG16",
            variable=self._modelo_nome,
            value="vgg",
            font=FONTE_CORPO,
        ).pack(anchor="w", padx=14, pady=2)
        ctk.CTkRadioButton(
            painel_esq,
            text="DenseNet121",
            variable=self._modelo_nome,
            value="densenet",
            font=FONTE_CORPO,
        ).pack(anchor="w", padx=14, pady=2)

        rotulo_secao(painel_esq, "HIPERPARÂMETROS")

        def _linha_param(pai, rotulo, var, de, ate, passos, fmt="{:.5f}"):
            f = ctk.CTkFrame(pai, fg_color="transparent")
            f.pack(fill="x", padx=12, pady=2)
            lbl = ctk.CTkLabel(f, text=rotulo, font=FONTE_PEQUENA, width=80, anchor="w")
            lbl.pack(side="left")
            val_lbl = ctk.CTkLabel(
                f, text=fmt.format(var.get()), font=FONTE_PEQUENA, width=60
            )
            val_lbl.pack(side="right")

            def _upd(v):
                val_lbl.configure(text=fmt.format(float(v)))

            sl = ctk.CTkSlider(
                f, from_=de, to=ate, number_of_steps=passos, variable=var, command=_upd
            )
            sl.pack(side="left", fill="x", expand=True, padx=4)
            return sl

        self._var_lr = ctk.DoubleVar(value=3e-4)
        self._var_wd = ctk.DoubleVar(value=1e-4)
        self._var_dropout = ctk.DoubleVar(value=0.5)
        self._var_epocas = ctk.IntVar(value=20)

        _linha_param(painel_esq, "LR", self._var_lr, 1e-5, 1e-2, 50, fmt="{:.5f}")
        _linha_param(painel_esq, "W.Decay", self._var_wd, 0, 1e-2, 50, fmt="{:.5f}")
        _linha_param(
            painel_esq, "Dropout", self._var_dropout, 0.1, 0.8, 14, fmt="{:.2f}"
        )
        _linha_param(painel_esq, "Épocas", self._var_epocas, 5, 50, 45, fmt="{:.0f}")

        rotulo_secao(painel_esq, "FINE-TUNING")
        self._var_descongelar = ctk.IntVar(value=0)
        ctk.CTkLabel(
            painel_esq, text="Blocos conv descongelados:", font=FONTE_PEQUENA
        ).pack(anchor="w", padx=14)
        ctk.CTkSlider(
            painel_esq, from_=0, to=3, number_of_steps=3, variable=self._var_descongelar
        ).pack(padx=12, fill="x")
        ctk.CTkLabel(
            painel_esq, text="0=só classifier · 1-3=blocos finais", font=FONTE_PEQUENA
        ).pack(anchor="w", padx=14, pady=(0, 6))

        rotulo_secao(painel_esq, "CONTROLES")
        self._btn_treinar = botao(painel_esq, "▶ Treinar", self._treinar)
        self._btn_treinar.pack(padx=12, fill="x")
        self._btn_parar = botao(
            painel_esq,
            "⏹ Parar",
            self._parar,
            fg_color="transparent",
            border_width=1,
            state="disabled",
        )
        self._btn_parar.pack(padx=12, pady=4, fill="x")
        botao(painel_esq, "⚡ Classificar Teste", self._classificar).pack(
            padx=12, fill="x"
        )
        linha_salvar = ctk.CTkFrame(painel_esq, fg_color="transparent")
        linha_salvar.pack(fill="x", padx=12, pady=4)
        botao(
            linha_salvar,
            "💾 Salvar",
            self._salvar,
            fg_color="transparent",
            border_width=1,
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        botao(
            linha_salvar,
            "📂 Carregar",
            self._carregar_modelo,
            fg_color="transparent",
            border_width=1,
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))

        # Progresso
        linha_prog = ctk.CTkFrame(painel_esq, fg_color="transparent")
        linha_prog.pack(fill="x", padx=12, pady=(4, 0))
        self._barra_treino = ctk.CTkProgressBar(linha_prog)
        self._barra_treino.set(0)
        self._barra_treino.pack(fill="x")
        linha_ep = ctk.CTkFrame(painel_esq, fg_color="transparent")
        linha_ep.pack(fill="x", padx=12)
        self._lbl_epoca = ctk.CTkLabel(
            linha_ep, text="—", font=FONTE_PEQUENA, anchor="w"
        )
        self._lbl_epoca.pack(side="left")
        self._lbl_tempo = ctk.CTkLabel(
            linha_ep, text="", font=FONTE_PEQUENA, anchor="e"
        )
        self._lbl_tempo.pack(side="right")

        # ── Painel direito: métricas + gráficos ──────────────────────────────
        painel_dir = ctk.CTkFrame(self, corner_radius=10)
        painel_dir.pack(side="left", fill="both", expand=True)

        # Métricas binárias
        rotulo_secao(painel_dir, "MÉTRICAS — BINÁRIO")
        linha_met = ctk.CTkFrame(painel_dir, fg_color="transparent")
        linha_met.pack(fill="x", padx=12, pady=(0, 6))
        nomes = ["Sensib.", "Especif.", "Precisão", "Acurácia", "F1"]
        self._cards_metrica = [CartaoMetrica(linha_met, n) for n in nomes]
        [
            c.pack(side="left", expand=True, fill="both", padx=2)
            for c in self._cards_metrica
        ]

        # ITEM 7: Cards de resumo do treinamento
        rotulo_secao(painel_dir, "RESUMO DO TREINAMENTO")
        linha_resumo = ctk.CTkFrame(painel_dir, fg_color="transparent")
        linha_resumo.pack(fill="x", padx=12, pady=(0, 6))
        self._card_treino_qtd = CartaoMetrica(linha_resumo, "Imgs Treino")
        self._card_teste_qtd = CartaoMetrica(linha_resumo, "Imgs Teste")
        self._card_tempo_total = CartaoMetrica(linha_resumo, "Tempo Total")
        self._card_melhor_acc = CartaoMetrica(linha_resumo, "Melhor Acc")
        self._card_melhor_f1 = CartaoMetrica(linha_resumo, "Melhor F1")
        for c in (
            self._card_treino_qtd,
            self._card_teste_qtd,
            self._card_tempo_total,
            self._card_melhor_acc,
            self._card_melhor_f1,
        ):
            c.pack(side="left", expand=True, fill="both", padx=2)

        # Abas internas: Matriz de Confusão | Gráficos
        self._abas_internas = ctk.CTkTabview(painel_dir)
        self._abas_internas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._abas_internas.add("📊 Matriz de Confusão")
        self._abas_internas.add("📈 Gráficos de Convergência")

        # Tab 1: Matriz de confusão 4 classes
        tab_matriz = self._abas_internas.tab("📊 Matriz de Confusão")
        rotulo_secao(tab_matriz, "MATRIZ — 4 CLASSES")
        area_mat = ctk.CTkFrame(tab_matriz, fg_color="transparent")
        area_mat.pack(padx=12)
        rotulos = ["I", "II", "III", "IV"]
        cab = ctk.CTkFrame(area_mat, fg_color="transparent")
        cab.pack()
        ctk.CTkLabel(cab, text="Pred→\nReal↓", font=FONTE_PEQUENA, width=70).pack(
            side="left"
        )
        for r in rotulos:
            ctk.CTkLabel(cab, text=f"B{r}", font=FONTE_PEQUENA, width=70).pack(
                side="left"
            )
        self._celulas_cm: list[list[ctk.CTkLabel]] = []
        for i, ri in enumerate(rotulos):
            linha_cm = ctk.CTkFrame(area_mat, fg_color="transparent")
            linha_cm.pack(pady=1)
            ctk.CTkLabel(linha_cm, text=f"B{ri}", font=FONTE_PEQUENA, width=70).pack(
                side="left"
            )
            row = []
            for j in range(4):
                cel = ctk.CTkLabel(
                    linha_cm,
                    text="—",
                    font=FONTE_CORPO,
                    corner_radius=4,
                    width=66,
                    height=30,
                    fg_color=("#d0e8ff" if i == j else "#f5f5f5"),
                )
                cel.pack(side="left", padx=2)
                row.append(cel)
            self._celulas_cm.append(row)
        linha_m4 = ctk.CTkFrame(tab_matriz, fg_color="transparent")
        linha_m4.pack(fill="x", padx=12, pady=(12, 6))
        self._card_sens_media = CartaoMetrica(linha_m4, "Sensib. Média")
        self._card_espec_media = CartaoMetrica(linha_m4, "Especif. Média")
        self._card_tempo_exec = CartaoMetrica(linha_m4, "Tempo")
        for c in (self._card_sens_media, self._card_espec_media, self._card_tempo_exec):
            c.pack(side="left", expand=True, fill="both", padx=3)

        # Tab 2: Gráficos de convergência
        tab_graf = self._abas_internas.tab("📈 Gráficos de Convergência")
        fundo_g = ctk.CTkFrame(tab_graf, corner_radius=6)
        fundo_g.pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_grafico = tk.Canvas(fundo_g, bg="#f8f8f8", highlightthickness=0)
        sv_g = ctk.CTkScrollbar(fundo_g, command=self._canvas_grafico.yview)
        sh_g = ctk.CTkScrollbar(
            fundo_g, orientation="horizontal", command=self._canvas_grafico.xview
        )
        self._canvas_grafico.configure(yscrollcommand=sv_g.set, xscrollcommand=sh_g.set)
        sh_g.pack(side="bottom", fill="x")
        sv_g.pack(side="right", fill="y")
        self._canvas_grafico.pack(fill="both", expand=True)
        self._lbl_grafico_hint = ctk.CTkLabel(
            tab_graf,
            text="Os gráficos aparecerão após o treinamento.",
            font=FONTE_PEQUENA,
        )
        self._lbl_grafico_hint.pack()

    # ── Treino ────────────────────────────────────────────────────────────────

    def _treinar(self):
        if not TORCH_OK:
            messagebox.showerror("Erro", "PyTorch não instalado.")
            return
        train_loader = self._aba_dataset.train_loader
        test_loader = self._aba_dataset.test_loader
        if not train_loader:
            messagebox.showwarning(
                "Aviso", "Carregue e processe o dataset antes de treinar."
            )
            return

        n_classes = 2 if self._modo.get() == "binario" else 4
        nome_modelo = self._modelo_nome.get()

        if nome_modelo == "vgg":
            self._modelo = ClassificadorVGG(
                n_classes=n_classes, dropout=self._var_dropout.get()
            )
        else:
            self._modelo = ClassificadorDenseNet(
                n_classes=n_classes, dropout=self._var_dropout.get()
            )

        n_blocos = int(self._var_descongelar.get())
        if n_blocos > 0:
            self._modelo.descongelar_ultimas_conv(n_blocos)

        self._parar_flag.clear()
        self._btn_treinar.configure(state="disabled")
        self._btn_parar.configure(state="normal")
        self._barra_treino.set(0)
        self._historico = {}

        n_epocas = int(self._var_epocas.get())
        threading.Thread(
            target=treinar_modelo,
            kwargs=dict(
                modelo=self._modelo,
                train_loader=train_loader,
                test_loader=test_loader,
                n_epocas=n_epocas,
                lr=self._var_lr.get(),
                weight_decay=self._var_wd.get(),
                modo=self._modo.get(),
                callback_epoca=self._atualizar_ui_epoca,
                callback_fim=self._treino_concluido,
                parar_flag=self._parar_flag,
            ),
            daemon=True,
        ).start()
        nome_exibicao = "VGG16" if self._modelo_nome.get() == "vgg" else "DenseNet121"
        self._status.definir(f"Treinando {nome_exibicao}…")

    def _parar(self):
        self._parar_flag.set()
        self._status.definir("Parando após a época atual…")

    def _atualizar_ui_epoca(self, info: dict):
        """Callback chamado ao fim de cada época — atualiza UI de forma thread-safe."""

        def _upd():
            ep = info["epoca"]
            total_ep = len(info["train_loss"])
            fracao = ep / max(total_ep, 1)
            loss_tr = info["train_loss"][-1]
            loss_val = info["val_loss"][-1]
            acc_val = info["val_acc"][-1]
            tempo = info["tempo_por_epoca"][-1]
            self._barra_treino.set(fracao)
            self._lbl_epoca.configure(
                text=f"Época {ep} | loss_tr={loss_tr:.4f} "
                f"| loss_val={loss_val:.4f} | acc_val={acc_val:.3f}"
            )
            self._lbl_tempo.configure(text=f"{tempo:.1f}s")
            if info.get("status") == "early_stop":
                self._status.definir(f"Early stop na época {ep}")

        self.after(0, _upd)

    def _treino_concluido(self, historico: dict):
        """Callback chamado ao fim do treinamento — gera gráficos."""
        self._historico = historico
        self.after(0, self._pos_treino)

    def _pos_treino(self):
        self._btn_treinar.configure(state="normal")
        self._btn_parar.configure(state="disabled")
        self._barra_treino.set(1.0)
        n_ep = self._historico.get("epocas_rodadas", 0)
        self._status.definir(f"Treino concluído — {n_ep} épocas.")

        # ITEM 7: preenche cards de resumo
        n_tr = (
            len(self._aba_dataset.train_loader.dataset)
            if self._aba_dataset.train_loader
            else 0
        )
        n_te = (
            len(self._aba_dataset.test_loader.dataset)
            if self._aba_dataset.test_loader
            else 0
        )
        tempo_total = self._historico.get("tempo_total", 0.0)
        melhor_acc = self._historico.get("melhor_acc", 0.0)
        melhor_f1 = self._historico.get("melhor_f1", 0.0)
        self._card_treino_qtd.definir(str(n_tr))
        self._card_teste_qtd.definir(str(n_te))
        self._card_tempo_total.definir(f"{tempo_total:.1f}s")
        self._card_melhor_acc.definir(f"{melhor_acc:.3f}")
        self._card_melhor_f1.definir(f"{melhor_f1:.3f}")

        _logger.info(
            f"Treino concluído | épocas={n_ep} | imgs_treino={n_tr} | "
            f"imgs_teste={n_te} | melhor_acc={melhor_acc:.3f} | "
            f"melhor_f1={melhor_f1:.3f} | tempo_total={tempo_total:.1f}s"
        )

        self._renderizar_graficos()

    def _renderizar_graficos(self):
        """Renderiza os gráficos de convergência no canvas da aba."""
        if not self._historico:
            return
        modo_str = "Binário" if self._modo.get() == "binario" else "4 Classes"
        nome_exibicao = "VGG16" if self._modelo_nome.get() == "vgg" else "DenseNet121"
        img_pil = gerar_graficos_convergencia(
            self._historico, titulo=f"{nome_exibicao} — {modo_str}"
        )
        if img_pil is None:
            return
        self._lbl_grafico_hint.pack_forget()
        self._canvas_grafico.update_idletasks()
        w = max(self._canvas_grafico.winfo_width(), 900)
        ratio = w / img_pil.width
        h = int(img_pil.height * ratio)
        img_pil_rs = img_pil.resize((w, h), Image.LANCZOS)
        self._img_grafico_tk = ImageTk.PhotoImage(img_pil_rs)
        self._canvas_grafico.delete("all")
        self._canvas_grafico.create_image(0, 0, anchor="nw", image=self._img_grafico_tk)
        self._canvas_grafico.configure(scrollregion=(0, 0, w, h))
        # Muda para a aba dos gráficos automaticamente
        self._abas_internas.set("📈 Gráficos de Convergência")

    # ── Classificação ─────────────────────────────────────────────────────────

    def _classificar(self):
        if not TORCH_OK:
            messagebox.showerror("Erro", "PyTorch não instalado.")
            return
        if self._modelo is None:
            messagebox.showwarning("Aviso", "Treine o modelo primeiro.")
            return
        test_loader = self._aba_dataset.test_loader
        if not test_loader:
            messagebox.showwarning("Aviso", "Carregue o dataset primeiro.")
            return
        threading.Thread(target=self._executar_classificacao, daemon=True).start()

    def _executar_classificacao(self):
        self.after(0, lambda: self._status.definir("Classificando conjunto de teste…"))
        _logger.info("Classificação iniciada no conjunto de teste")
        t0 = time.time()
        modelo = self._modelo.to(DEVICE)
        modelo.eval()
        modo = self._modo.get()
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in self._aba_dataset.test_loader:
                if modo == "binario":
                    labels = (labels >= 2).long()
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                saidas = modelo(imgs)
                preds = saidas.argmax(dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        decorrido = time.time() - t0

        # Métricas binárias
        # CORREÇÃO (Item 6): a mesma regra do treino — positivo = classes >= 2 (III+IV)
        # Antes estava ">= 1" para modo quadriclasse, o que incluía erroneamente
        # BI-RADS II (classe 1) como positivo, divergindo do critério de treino.
        met_bin = calcular_metricas_binario(
            [1 if v >= 2 else 0 for v in y_true] if modo != "binario" else y_true,
            [1 if v >= 2 else 0 for v in y_pred] if modo != "binario" else y_pred,
        )

        # Métricas 4 classes
        n_c = 2 if modo == "binario" else 4
        met4 = calcular_metricas_4classes(y_true, y_pred, n_classes=n_c)

        _logger.info(
            f"Classificação concluída | tempo={decorrido:.2f}s | "
            f"acc={met_bin['acuracia']:.3f} | sensib={met_bin['sensibilidade']:.3f} | "
            f"espec={met_bin['especificidade']:.3f} | F1={met_bin['f1']:.3f}"
        )

        self.after(0, lambda: self._exibir_resultados(met_bin, met4, decorrido))

    def _exibir_resultados(self, met_bin, met4, tempo):
        vals = [
            met_bin["sensibilidade"],
            met_bin["especificidade"],
            met_bin["precisao"],
            met_bin["acuracia"],
            met_bin["f1"],
        ]
        for cartao, v in zip(self._cards_metrica, vals):
            cartao.definir(f"{v:.3f}")

        # Matriz de confusão
        matriz = met4["matriz_confusao"]
        n_c = len(matriz)
        for i in range(4):
            for j in range(4):
                if i < n_c and j < n_c:
                    self._celulas_cm[i][j].configure(text=str(matriz[i][j]))
                else:
                    self._celulas_cm[i][j].configure(text="—")

        self._card_sens_media.definir(f"{met4['sensibilidade_media']:.3f}")
        self._card_espec_media.definir(f"{met4['especificidade_media']:.3f}")
        self._card_tempo_exec.definir(f"{tempo:.2f}s")
        self._status.definir("Classificação concluída.")
        self._abas_internas.set("📊 Matriz de Confusão")

    # ── Salvar ────────────────────────────────────────────────────────────────

    def _salvar(self):
        if self._modelo is None:
            messagebox.showwarning("Aviso", "Treine o modelo primeiro.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".pth", filetypes=[("PyTorch", "*.pth"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        try:
            torch.save(self._modelo.state_dict(), caminho)
            _logger.info(f"Modelo salvo: {caminho}")
            # Salva também o histórico em JSON ao lado do .pth
            hist_path = caminho.replace(".pth", "_historico.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(self._historico, f, indent=2, ensure_ascii=False)
            # Salva gráfico em PNG ao lado do .pth
            if self._historico:
                graf_path = caminho.replace(".pth", "_graficos.png")
                nome_exibicao = "VGG16" if self._modelo_nome.get() == "vgg" else "DenseNet121"
                modo_str = "Binário" if self._modo.get() == "binario" else "4 Classes"
                img_pil = gerar_graficos_convergencia(self._historico, titulo=f"{nome_exibicao} — {modo_str}")
                if img_pil:
                    img_pil.save(graf_path, dpi=(150, 150))
            messagebox.showinfo("Salvo", f"Modelo: {caminho}\nHistórico: {hist_path}")
            self._status.definir(f"Modelo salvo: {os.path.basename(caminho)}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    # ── Acesso externo (AbaGradCAM) ───────────────────────────────────────────
    @property
    def modelo(self):
        return self._modelo

    # ── Carregar modelo treinado ──────────────────────────────────────────────

    def _carregar_modelo(self):
        """
        ITEM 5: Carrega pesos de um arquivo .pth salvo previamente.

        Fluxo:
        1. Abre diálogo para selecionar o arquivo .pth
        2. Detecta automaticamente a arquitetura e o número de classes
           a partir das chaves do state_dict
        3. Instancia o modelo correto (VGG ou DenseNet, 2 ou 4 classes)
        4. Carrega os pesos com map_location (funciona em CPU e GPU)
        5. Disponibiliza o modelo para classificação e Grad-CAM imediatamente
        6. Tenta carregar histórico JSON correspondente (se existir)
        """
        if not TORCH_OK:
            messagebox.showerror("Erro", "PyTorch não instalado.")
            return
        caminho = filedialog.askopenfilename(
            filetypes=[("PyTorch weights", "*.pth"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        try:
            state = torch.load(caminho, map_location=DEVICE)

            # ── Detecta arquitetura pelo formato das chaves do state_dict ──
            chaves = list(state.keys())
            # VGG: chaves começam com "features.0.weight" (índice inteiro)
            # DenseNet: chaves começam com "features.conv0.weight"
            eh_densenet = any("denseblock" in k or "conv0" in k for k in chaves)

            # Detecta n_classes pelo tamanho da última camada do classifier
            ultima_chave = [k for k in chaves if "classifier" in k and "weight" in k]
            if ultima_chave:
                n_classes = state[ultima_chave[-1]].shape[0]
            else:
                n_classes = 4  # fallback

            # ── Instancia o modelo correto ─────────────────────────────────
            if eh_densenet:
                novo_modelo = ClassificadorDenseNet(n_classes=n_classes)
                self._modelo_nome.set("densenet")
            else:
                novo_modelo = ClassificadorVGG(n_classes=n_classes)
                self._modelo_nome.set("vgg")

            novo_modelo.load_state_dict(state)
            novo_modelo = novo_modelo.to(DEVICE)
            novo_modelo.eval()
            self._modelo = novo_modelo

            # Sincroniza o seletor de modo com n_classes detectado
            self._modo.set("binario" if n_classes == 2 else "quadriclasse")

            arq_nome = os.path.basename(caminho)
            tipo_str = "DenseNet121" if eh_densenet else "VGG16"
            modo_str = "Binário (2 classes)" if n_classes == 2 else "4 Classes"
            _logger.info(
                f"Modelo carregado: {caminho} | arch={tipo_str} | classes={n_classes}"
            )

            # Tenta carregar histórico JSON correspondente
            hist_path = caminho.replace(".pth", "_historico.json")
            if os.path.isfile(hist_path):
                with open(hist_path, "r", encoding="utf-8") as f:
                    self._historico = json.load(f)
                self._renderizar_graficos()
                _logger.info(f"Histórico carregado: {hist_path}")

            msg = (
                f"Modelo carregado com sucesso.\n\n"
                f"Arquivo : {arq_nome}\n"
                f"Arquitetura : {tipo_str}\n"
                f"Modo : {modo_str}\n\n"
                f"Pronto para Classificar e Grad-CAM."
            )
            messagebox.showinfo("Modelo carregado", msg)
            self._status.definir(
                f"Modelo carregado: {arq_nome} ({tipo_str}, {modo_str})"
            )
        except Exception as e:
            _logger.error(f"Erro ao carregar modelo: {e}")
            messagebox.showerror("Erro ao carregar modelo", str(e))


# =============================================================================
# ABA GRAD-CAM REAL
# =============================================================================


class AbaGradCAM(ctk.CTkFrame):
    """
    Aba de visualização Grad-CAM com implementação real via hooks PyTorch.

    Gera mapas de calor sobre a imagem original mostrando quais regiões
    da mama influenciaram a decisão da rede para a classe predita.
    """

    ROTULOS = {0: "BI-RADS I", 1: "BI-RADS II", 2: "BI-RADS III", 3: "BI-RADS IV"}

    def __init__(self, pai, status, aba_classif: AbaClassificacao):
        super().__init__(pai, fg_color="transparent")
        self._status = status
        self._aba_classif = aba_classif
        self._caminho = ""
        self._pil_orig = None
        self._grad_cam = None
        self._construir()

    def _construir(self):
        painel = ctk.CTkFrame(self, width=240, corner_radius=10)
        painel.pack(side="left", fill="y", padx=(0, 6))
        painel.pack_propagate(False)

        rotulo_secao(painel, "IMAGEM")
        botao(painel, "📂 Abrir Imagem", self._abrir).pack(padx=12, fill="x")
        self._lbl_nome = ctk.CTkLabel(
            painel, text="—", font=FONTE_PEQUENA, wraplength=200, anchor="w"
        )
        self._lbl_nome.pack(padx=12, pady=4, anchor="w")

        rotulo_secao(painel, "PRÉ-PROCESSAMENTO")
        self._var_segmentar = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            painel,
            text="Segmentar antes da classificação",
            variable=self._var_segmentar,
            font=FONTE_PEQUENA,
        ).pack(anchor="w", padx=14, pady=2)

        rotulo_secao(painel, "RESULTADO")
        self._card_classe = CartaoMetrica(painel, "BIRADS Predito")
        self._card_confianca = CartaoMetrica(painel, "Confiança")
        self._card_classe.pack(padx=12, fill="x")
        self._card_confianca.pack(padx=12, fill="x", pady=4)

        botao(painel, "🔥 Gerar Grad-CAM", self._gerar).pack(padx=12, fill="x")
        botao(
            painel,
            "💾 Salvar Resultado",
            self._salvar_resultado,
            fg_color="transparent",
            border_width=1,
        ).pack(padx=12, pady=4, fill="x")

        rotulo_secao(painel, "LEGENDA GRAD-CAM")
        ctk.CTkLabel(
            painel,
            text="Azul → baixa ativação\n"
            "Verde → ativação média\n"
            "Amarelo → ativação alta\n"
            "Vermelho → máxima ativação",
            font=FONTE_PEQUENA,
            justify="left",
            anchor="w",
        ).pack(padx=14, anchor="w")

        rotulo_secao(painel, "INFO")
        self._info_cam = ctk.CTkTextbox(
            painel, height=80, font=FONTE_MONO, state="disabled"
        )
        self._info_cam.pack(padx=12, fill="x")

        # Área de visualização
        area = ctk.CTkFrame(self, corner_radius=10)
        area.pack(side="left", fill="both", expand=True)
        cab = ctk.CTkFrame(area, fg_color="transparent", height=30)
        cab.pack(fill="x", padx=12, pady=(8, 4))
        cab.pack_propagate(False)
        ctk.CTkLabel(
            cab, text="Original (segmentada)", font=FONTE_SECAO, anchor="w"
        ).pack(side="left", expand=True)
        ctk.CTkLabel(cab, text="Grad-CAM Overlay", font=FONTE_SECAO, anchor="w").pack(
            side="left", expand=True
        )

        area_canvas = ctk.CTkFrame(area, corner_radius=6, fg_color="#e8e8e8")
        area_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._canvas_orig = tk.Canvas(area_canvas, bg="#e8e8e8", highlightthickness=0)
        self._canvas_cam = tk.Canvas(area_canvas, bg="#e8e8e8", highlightthickness=0)
        self._canvas_orig.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self._canvas_cam.pack(side="left", fill="both", expand=True, padx=(2, 0))

    def _atualizar_info(self, texto: str):
        self._info_cam.configure(state="normal")
        self._info_cam.delete("1.0", "end")
        self._info_cam.insert("1.0", texto)
        self._info_cam.configure(state="disabled")

    def _abrir(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        self._caminho = caminho
        self._lbl_nome.configure(text=os.path.basename(caminho))
        try:
            img = Image.open(caminho)
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(
                    np.uint8
                )
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            self._pil_orig = Image.fromarray(arr).convert("L")
            self._exibir_canvas(self._canvas_orig, self._pil_orig, "_tk_orig")
            self._status.definir(f"Imagem: {os.path.basename(caminho)}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _exibir_canvas(self, canvas, img, attr):
        canvas.update_idletasks()
        larg = max(canvas.winfo_width(), 200)
        alt = max(canvas.winfo_height(), 200)
        copia = img.copy()
        copia.thumbnail((larg, alt), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(copia)
        setattr(self, attr, img_tk)
        canvas.delete("all")
        canvas.create_image(larg // 2, alt // 2, anchor="center", image=img_tk)

    def _gerar(self):
        if not TORCH_OK:
            messagebox.showerror("Erro", "PyTorch não instalado.")
            return
        if not self._caminho:
            messagebox.showwarning("Aviso", "Selecione uma imagem.")
            return
        modelo = self._aba_classif.modelo
        if modelo is None:
            messagebox.showwarning(
                "Aviso", "Treine ou carregue um modelo na aba Classificação."
            )
            return
        threading.Thread(target=self._executar_gradcam, daemon=True).start()

    def _executar_gradcam(self):
        self.after(0, lambda: self._status.definir("Gerando Grad-CAM…"))
        _logger.info(f"Grad-CAM iniciado: {os.path.basename(self._caminho)}")

        modelo = self._aba_classif.modelo
        modelo = modelo.to(DEVICE).eval()

        # Pré-processa a imagem (segmentação opcional + normalização ImageNet)
        img_pil = self._pil_orig
        if self._var_segmentar.get():
            img_pil = segmentar_mama(img_pil.convert("L"))
            img_pil = recortar_bounding_box(img_pil)
        img_rgb = img_pil.convert("RGB").resize((224, 224), Image.LANCZOS)

        transf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = transf(img_rgb).unsqueeze(0)  # [1, 3, 224, 224]

        # Grad-CAM real
        if self._grad_cam is None or True:  # recria sempre (modelo pode ter mudado)
            self._grad_cam = GradCAM(modelo)

        mapa_np = self._grad_cam.gerar(tensor)  # [224, 224] float32 [0,1]

        # Predição e confiança
        with torch.no_grad():
            logits = modelo(tensor.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[0]
            classe_idx = probs.argmax().item()
            confianca = probs[classe_idx].item()

        # Overlay colorido (blend 55% CAM + 45% original)
        cam_rgb = Image.fromarray(aplicar_colormap_jet(mapa_np))
        orig_rgb = img_rgb.copy()
        misturado = Image.blend(orig_rgb, cam_rgb, alpha=0.55)

        rotulo = self.ROTULOS.get(classe_idx, str(classe_idx))
        n_classes = logits.shape[1]
        info = (
            f"Classe predita: {rotulo}\n"
            f"Confiança: {confianca:.1%}\n"
            f"Modo: {'binário' if n_classes == 2 else '4 classes'}\n"
            f"Camada alvo: {self._grad_cam.nome_camada_alvo}"
        )

        self.after(
            0,
            lambda m=misturado, r=rotulo, c=confianca, i=info: self._exibir_resultado(
                m, r, c, i
            ),
        )

    def _exibir_resultado(self, misturado, rotulo, confianca, info):
        self._exibir_canvas(self._canvas_cam, misturado, "_tk_cam")
        self._card_classe.definir(rotulo)
        self._card_confianca.definir(f"{confianca:.1%}")
        self._atualizar_info(info)
        _logger.info(f"Grad-CAM gerado | classe={rotulo} | confiança={confianca:.1%}")
        self._status.definir("Grad-CAM gerado.")

    def _salvar_resultado(self):
        """Salva o overlay Grad-CAM em PNG."""
        if not hasattr(self, "_tk_cam"):
            messagebox.showwarning("Aviso", "Gere o Grad-CAM primeiro.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        try:
            # Re-gera a imagem misturada em tamanho original
            modelo = self._aba_classif.modelo
            img_pil = self._pil_orig
            if self._var_segmentar.get():
                img_pil = segmentar_mama(img_pil.convert("L"))
                img_pil = recortar_bounding_box(img_pil)
            img_rgb = img_pil.convert("RGB").resize((224, 224), Image.LANCZOS)
            transf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            tensor = transf(img_rgb).unsqueeze(0)
            mapa_np = self._grad_cam.gerar(tensor)
            cam_rgb = Image.fromarray(aplicar_colormap_jet(mapa_np))
            misturado = Image.blend(img_rgb, cam_rgb, alpha=0.55)
            misturado.save(caminho)
            messagebox.showinfo("Salvo", f"Grad-CAM salvo em:\n{caminho}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))


# =============================================================================
# APLICAÇÃO PRINCIPAL
# =============================================================================


class AplicacaoMamografia(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MamoVision — Segmentação e Classificação Mamográfica")
        self.geometry("1380x820")
        self.minsize(1000, 680)
        self._construir()

    def _construir(self):
        cabecalho = ctk.CTkFrame(self, corner_radius=0, height=52)
        cabecalho.pack(fill="x")
        cabecalho.pack_propagate(False)
        ctk.CTkLabel(cabecalho, text="MamoVision", font=FONTE_TITULO).pack(
            side="left", padx=12
        )
        ctk.CTkLabel(
            cabecalho,
            text="Segmentação e Classificação Mamográfica · PUC Minas · VGG16 + DenseNet121",
            font=FONTE_PEQUENA,
        ).pack(side="left")
        lbl_device = ctk.CTkLabel(
            cabecalho,
            text=f"Device: {DEVICE}" if DEVICE else "PyTorch indisponível",
            font=FONTE_PEQUENA,
        )
        lbl_device.pack(side="right", padx=12)

        self._barra_status = BarraStatus(self)
        self._barra_status.pack(fill="x", side="bottom")

        abas = ctk.CTkTabview(self)
        abas.pack(fill="both", expand=True, padx=10, pady=6)

        # Cria as abas em ordem — AbaDataset é referenciada pelas seguintes
        abas.add("📷 Visualizador")
        abas.add("📦 Dataset")
        abas.add("🧠 Classificação")
        abas.add("🔥 Grad-CAM")

        AbaVisualizador(abas.tab("📷 Visualizador"), self._barra_status).pack(
            fill="both", expand=True
        )

        self._aba_dataset = AbaDataset(abas.tab("📦 Dataset"), self._barra_status)
        self._aba_dataset.pack(fill="both", expand=True)

        self._aba_classif = AbaClassificacao(
            abas.tab("🧠 Classificação"), self._barra_status, self._aba_dataset
        )
        self._aba_classif.pack(fill="both", expand=True)

        AbaGradCAM(abas.tab("🔥 Grad-CAM"), self._barra_status, self._aba_classif).pack(
            fill="both", expand=True
        )


def main():
    app = AplicacaoMamografia()
    app.mainloop()


if __name__ == "__main__":
    main()
