# MamoVision - Segmentação e Classificação de Imagens Mamográficas

## 📋 Descrição

**MamoVision** é uma aplicação educacional desenvolvida em Python que implementa técnicas de processamento e análise de imagens para segmentação e classificação de imagens mamográficas. O projeto utiliza aprendizado de máquina para auxiliar na detecção de diferentes graus de densidade mamária segundo a escala BI-RADS.

**Disciplina:** Processamento e Análise de Imagens (PAI)  
**Instituição:** PUC Minas  
**Professor:** Alexei Machado

---

## ✨ Funcionalidades Principais

### 📷 1. Visualizador de Imagens
- Carregar imagens mamográficas em formatos PNG e TIFF
- Zoom ajustável (20% a 400%)
- Visualização detalhada das propriedades da imagem
- Exibição de máscara de segmentação em overlay

### 📦 2. Gerenciamento de Dataset
- Carregamento automático de datasets de imagens
- Classificação automática por classes BI-RADS (I, II, III, IV)
- Split automático entre conjunto de treino e teste
- Estatísticas de distribuição de classes
- **Data Augmentation**: aumento de dados através de rotações (-20° a +20°)

### 🧠 3. Classificação com Deep Learning
- Modo binário (Low Density vs High Density)
- Modo 4-classes (BI-RADS I, II, III, IV)
- Treinamento de modelos com progresso em tempo real
- Matriz de confusão interativa
- Métricas detalhadas: sensibilidade, especificidade, precisão, acurácia, F1-Score

### 🔥 4. Interpretabilidade - Grad-CAM
- Visualização de regiões de ativação da rede neural
- Heatmap colorido indicando áreas de alta ativação
- Overlay da visualização sobre a imagem original
- Predição de classe com nível de confiança

---

## 🛠️ Tecnologias

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| `customtkinter` | ≥0.6 | Interface gráfica moderna |
| `tkinter` | - | Framework base de GUI |
| `Pillow (PIL)` | ≥9.0 | Processamento de imagens |
| `NumPy` | ≥1.20 | Computação numérica |
| `PyTorch` | ≥1.9 | Deep Learning (opcional) |
| `scikit-learn` | ≥1.0 | Métricas e pré-processamento |

---

## 📦 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes)

### Passo 1: Clonar ou extrair o repositório
```bash
cd "Trabalho Final PAI"
```

### Passo 2: Criar ambiente virtual (recomendado)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Passo 3: Instalar dependências
```bash
pip install -r requirements.txt
```

Ou manualmente:
```bash
pip install customtkinter pillow numpy scikit-learn torch torchvision
```

### Passo 4: Executar a aplicação
```bash
python main600linhas.py
```

---

## 📂 Estrutura do Projeto

```
Trabalho Final PAI/
├── main600linhas.py          # Arquivo principal da aplicação
├── README.md                 # Este arquivo
├── requirements.txt          # Dependências do projeto
└── data/                     # Diretório com datasets
    ├── D + left + MLO/      # BI-RADS I (Gordura)
    ├── E - left - MLO/      # BI-RADS II (Fibroglandular)
    ├── F + left + MLO/      # BI-RADS III (Denso-Heterogêneo)
    └── G + left + MLO/      # BI-RADS IV (Extremamente Denso)
```

---

## 🚀 Como Usar

### 1️⃣ Visualizador
1. Abra a aba **"📷 Visualizador"**
2. Clique em **"📂 Abrir PNG / TIFF"**
3. Selecione uma imagem mamográfica
4. Use o controle de zoom para ajustar visualização
5. Clique em **"⚙ Segmentar Mama"** para extrair a região de interesse
6. Alterne entre imagem original e máscara com **"👁 Mostrar Máscara"**

### 2️⃣ Dataset
1. Abra a aba **"📦 Dataset"**
2. Clique em **"📁 Selecionar Diretório"** e escolha a pasta com imagens
3. Observe a distribuição de classes no gráfico
4. Clique em **"⟳ Realizar Aumento"** para gerar variações das imagens
5. Monitore o progresso e estatísticas

### 3️⃣ Classificação
1. Abra a aba **"🧠 Classificação"**
2. Escolha o modo: **Binário** ou **4 Classes**
3. Clique em **"▶ Treinar Modelo"** para iniciar o treinamento
4. Monitore as épocas e tempo de execução
5. Clique em **"⚡ Classificar Teste"** para avaliar o modelo
6. Visualize métricas e matriz de confusão
7. Clique em **"💾 Salvar Modelo"** para persister

### 4️⃣ Grad-CAM
1. Abra a aba **"🔥 Grad-CAM"**
2. Clique em **"📂 Abrir Imagem"** para selecionar
3. Clique em **"🔥 Gerar Grad-CAM"**
4. Compare a visualização original com o heatmap
5. Observe a classe predita e nível de confiança

---

## 📊 Escala BI-RADS

| Classe | Código | Descrição | Densidade |
|--------|--------|-----------|-----------|
| **I** | D | Gordura | Baixa |
| **II** | E | Fibroglandular | Média-Baixa |
| **III** | F | Denso-Heterogêneo | Média-Alta |
| **IV** | G | Extremamente Denso | Alta |

---

## 🎨 Tema de Cores

A aplicação suporta dois temas:
- **Tema Escuro**: Cores em tons azuis e cinzentos para ambiente com baixa luminosidade
- **Tema Claro**: Cores suaves em tons claros para ambiente bem iluminado

Alterne entre temas clicando no botão **"☀ Tema Claro / 🌙 Tema Escuro"** no canto superior direito.

---

## 📈 Métricas de Classificação

### Binária
- **Sensibilidade**: Taxa de verdadeiros positivos
- **Especificidade**: Taxa de verdadeiros negativos
- **Precisão**: Proporção de predições corretas positivas
- **Acurácia**: Taxa geral de acertos
- **F1-Score**: Média harmônica de precisão e sensibilidade

### Multi-classe (BI-RADS)
- **Matriz de Confusão**: Visualiza erros por classe
- **Sensibilidade Média**: Média de sensibilidades por classe
- **Especificidade Média**: Média de especificidades por classe
- **Tempo de Execução**: Duração da classificação

---

## 🔧 Personalização

### Ajustar cores
Edite as paletas no início do arquivo:
```python
COLORS_DARK = {...}   # Cores do tema escuro
COLORS_LIGHT = {...}  # Cores do tema claro
```

### Modificar parâmetros de segmentação
Veja o método `_run_segmentation()` para ajustar:
- Limiar de Otsu
- Tamanho do kernel de erosão/dilatação
- Parâmetros de filtragem

### Alterar augmentation
Edite a lista de ângulos em `_run_augment()`:
```python
angles = [-20, -10, 0, 10, 20]  # Customize aqui
```

---

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| Erro ao importar customtkinter | Execute `pip install customtkinter` |
| Imagem não carrega | Verifique formato (PNG, TIFF) e caminho |
| Interface lenta | Reduza resolução das imagens |
| Modelo não treina | Verifique se dataset foi carregado corretamente |

---

## 📝 Notas Importantes

- Este é um projeto **educacional** para fins de aprendizado
- As implementações de treino e classificação contêm **placeholders** que devem ser substituídos por lógica real com PyTorch/TensorFlow
- Para uso em produção, é necessário validação clínica adequada
- Sensibilidade: dados reais de mamografia contêm informações pessoais

---

## 👥 Contribuidores

Grupo de Trabalho Final PAI - PUC Minas  
*[Preencher com nomes, matrículas, cursos e campus dos integrantes]*

---

## 📄 Licença

Este projeto é fornecido para fins educacionais.

---

## 📚 Referências

- BIRADS (Breast Imaging-Reporting and Data System)
- Otsu's Binarization Algorithm
- Grad-CAM: Visual Explanations from Deep Networks
- Deep Learning para Análise de Imagens Médicas

---

## 💬 Suporte

Para dúvidas ou problemas, consulte:
- Prof. Alexei Machado
- Documentação do CustomTkinter: https://github.com/TomSchimansky/CustomTkinter
- Documentação PIL/Pillow: https://pillow.readthedocs.io/

---

**Última atualização:** Maio 2026
