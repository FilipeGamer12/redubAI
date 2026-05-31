# RedubAI

RedubAI é um script em Python para redublagem automática de vídeos MKV com IA.

Ele faz o seguinte fluxo:

1. Extrai o áudio do vídeo.
2. Transcreve as falas com Whisper.
3. Gera um SRT original.
4. Traduz o SRT com Ollama.
5. Gera TTS com `edge-tts`.
6. Sincroniza melhor os trechos de fala.
7. Faz mixagem com “ducking” do áudio original durante a fala.
8. Exporta um MKV final com **uma única trilha de áudio**, dublagem estilo documentário.

## Funcionalidades

- Entrada obrigatória em **MKV**
- Transcrição automática com Whisper
- Tradução local com Ollama
- Geração de voz com `edge-tts`
- Ajuste de duração dos trechos de TTS para caber no segmento
- Mixagem do áudio original com redução de volume enquanto o TTS fala
- Suporte a CPU e GPU compatível com PyTorch
- Opção de reutilizar SRT original e/ou SRT traduzido

## Requisitos

### Sistema

- **Python 3.10+** recomendado
- **FFmpeg** instalado e disponível no PATH
- **FFprobe** instalado e disponível no PATH
- **Ollama** em execução localmente
- Um modelo de tradução baixado no Ollama

### Dependências Python

Instale com:

```bash
pip install openai-whisper edge-tts pydub torch
```

Dependências diretas usadas pelo projeto:

- `openai-whisper`
- `torch`
- `edge-tts`
- `pydub`

Dependências indiretas comuns que podem ser instaladas junto dos pacotes acima:

- `numpy`
- `tqdm`
- `more-itertools`
- `regex`

Essas dependências indiretas normalmente são puxadas automaticamente pelo `pip`, então não precisam ser instaladas manualmente na maioria dos casos.

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/FilipeGamer12/redubAI.git
cd redubAI
```

### 2. Criar e ativar ambiente virtual

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências Python

```bash
pip install --upgrade pip
pip install openai-whisper edge-tts pydub torch
```

### 4. Instalar FFmpeg

#### Windows

Baixe e instale o FFmpeg e adicione a pasta `bin` ao PATH.

#### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Arch Linux

```bash
sudo pacman -S ffmpeg
```

### 5. Instalar e iniciar o Ollama

Baixe o Ollama para o seu sistema e depois inicie o servidor local.

```bash
ollama serve
```

Depois, baixe o modelo que será usado na tradução:

```bash
ollama pull gemma2:9b
```

Você pode trocar o modelo por outro configurado no script, como por exemplo:

- `gemma2:9b`
- `qwen2.5:7b`
- outro modelo compatível com Ollama

## Uso

```bash
python redubAI.py --input input/video.mkv --output output/video_dublado.mkv
```

### Parâmetros principais

- `--input` / `-i`: caminho do vídeo MKV de entrada
- `--output` / `-o`: caminho do MKV de saída
- `--asr-model`: modelo Whisper usado na transcrição
- `--qwen-model`: modelo do Ollama usado na tradução
- `--ollama-url`: URL do servidor Ollama
- `--voice`: voz usada no `edge-tts`
- `--device`: `cpu` ou `cuda`
- `--keep-temp`: mantém arquivos temporários para depuração

### Reutilizar SRT existente

Se você já tiver os arquivos gerados, pode pular etapas:

```bash
python redubAI.py   --input input/video.mkv   --output output/video_dublado.mkv   --skip-transcription   --original-srt original.srt   --skip-translation   --translated-srt translated.srt
```

## Como funciona a mixagem de áudio

O projeto não adiciona uma segunda faixa separada para o vídeo final.

Em vez disso:

- quando o TTS está falando, o áudio original é reduzido fortemente;
- quando não há fala do TTS, o áudio original volta ao volume normal;
- o resultado é uma **única faixa de áudio** com sensação de dublagem sobreposta, no estilo documentário.

## Arquivos gerados

Durante a execução, o script pode gerar:

- `original.srt`
- `translated.srt`
- áudio temporário do TTS
- faixa de áudio final em `m4a`
- vídeo final em `mkv`

## Observações importantes

- O vídeo de entrada precisa ser **MKV**.
- O FFmpeg precisa estar acessível no PATH.
- O Ollama precisa estar rodando antes de iniciar a tradução.
- O modelo de tradução precisa estar baixado no Ollama.
- Em GPUs AMD, o suporte depende de uma instalação compatível do PyTorch/ROCm.
- Em CPU, o processo funciona, mas a transcrição pode ser mais lenta.

## Solução de problemas

### “ffmpeg não encontrado”
Instale o FFmpeg e confirme se `ffmpeg` e `ffprobe` funcionam no terminal.

### “Falha ao acessar o Ollama”
Verifique se o serviço está ativo:

```bash
ollama serve
```

E se o modelo foi baixado:

```bash
ollama list
```

### Whisper muito lento
Use um modelo menor, como:

```bash
--asr-model base
```

ou execute em GPU com:

```bash
--device cuda
```

### O TTS fica adiantado ou desalinhado
O script já faz refinamento de timing, mas em vídeos com fala muito rápida ou cortes bruscos, vale testar:

- um modelo Whisper melhor;
- um vídeo com áudio limpo;
- um modelo de tradução mais curto e natural.