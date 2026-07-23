# Modelo Treinado SEFAZ-RN

Este projeto é uma solução completa em **Python** voltada para a automação do fluxo de **coleta, rotulagem, treinamento e avaliação de modelos de Machine Learning/Deep Learning** para o contexto de dados e captchas/recursos do sistema da SEFAZ-RN.

---

## 📌 Funcionalidades

* **Coleta de Dados (`coletor.py`):** Script para automação no download e extração de amostras do dataset.
* **Rotulagem de Amostras (`rotular.py`):** Ferramenta auxiliar para categorização e anotação dos dados coletados.
* **Validação do Dataset (`verificar_dataset.py`):** Checagem de integridade, consistência e limpeza dos dados de treino/teste.
* **Treinamento do Modelo (`treinar.py`):** Pipeline de treinamento e ajuste fino de redes de aprendizado de máquina (utilizando a biblioteca `dddd_trainer` / arquiteturas customizadas).
* **Avaliação de Desempenho (`acuracia.py`):** Cálculo e geração de relatórios de precisão e desempenho do modelo gerado.
* **Execução/Predição (`varrer.py`):** Processamento em lote para inferência automatizada.
* **API / Aplicação (`app.py`):** Interface de serviço/API para disponibilizar o modelo para consumo por outras aplicações.

---

## 📁 Estrutura do Repositório

```text
.
├── configurações/            # Arquivos de configuração do ambiente e treinamento
├── dataset_recife/          # Subconjunto ou conjunto de dados de treino rotulados
├── dddd_trainer/            # Módulo de treinamento e arquiteturas de rede
├── modelo/                  # Pesos e artefatos do modelo treinado gerado
├── redes/                   # Definições de arquiteturas de redes neurais
├── projetos/                # Configurações e subprojetos específicos (ex: SEFAZ-RN)
├── ferramentas/             # Utilitários e rotinas auxiliares
├── utilitários/             # Módulos helpers do sistema
├── acuracia.py              # Script de cálculo de métricas e acurácia
├── app.py                   # Ponto de entrada da aplicação / API
├── coletor.py               # Script de raspagem e coleta de imagens/dados
├── rotular.py               # Interface/script para rotulagem dos dados
├── treinar.py               # Pipeline principal de treinamento
├── varrer.py                # Script para varredura e predições em massa
├── verificar_dataset.py     # Script para auditoria do dataset
├── requisitos.txt           # Dependências do projeto em Python
└── LICENÇA                  # Licença Apache 2.0

```

---

## 🚀 Como Executar

### 1. Pré-requisitos

Certifique-se de ter o **Python 3.8+** instalado em seu sistema.

### 2. Instalação

Clone o repositório e instale as dependências necessárias:

```bash
# Clonar o repositório
git clone https://github.com/soouzaana/modelo-treinado-sefaz-rn.git

# Acessar a pasta do projeto
cd modelo-treinado-sefaz-rn

# Criar e ativar um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate

# Instalar as dependências
pip install -r requisitos.txt

```

---

## 🛠️ Modo de Uso

### 1. Coleta e Preparação dos Dados

```bash
python coletor.py
python verificar_dataset.py

```

### 2. Treinamento do Modelo

```bash
python treinar.py

```

### 3. Avaliação da Acurácia

```bash
python acuracia.py

```

### 4. Execução da Aplicação / API

```bash
python app.py

```

---

## 📜 Licença

Este projeto está sob a licença **Apache-2.0**. Consulte o arquivo [LICENÇA](https://www.google.com/search?q=LICEN%C3%87A) para obter mais detalhes.

---

## 👩‍💻 Autora

Desenvolvido por **Ana Maria Brito Souza** ([@soouzaana](https://github.com/soouzaana)).
