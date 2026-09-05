# Установка

Проект рассчитан на Python 3.12. Основные зависимости приложения находятся в
`requirements.txt` в корне проекта.

```powershell
python -m pip install -r requirements.txt
```

PyTorch устанавливается отдельно, потому что команда зависит от видеокарты и
режима работы. Используйте официальный установщик PyTorch и выбранную им
команду для Windows/CPU или нужной CUDA-сборки:

```powershell
python -m pip install torch torchvision
```

Важно для RTX 5070 Laptop: не используйте старую сборку `torch==2.6.0+cu124`.
В текущем окружении она видит видеокарту, но не поддерживает ее CUDA-архитектуру
`sm_120`. Для текущего приложения SMT принудительно работает на CPU, поэтому
LM Studio может оставить GPU целиком основной Qwen-модели.

В проекте уже находится локальная копия архитектуры SMT в каталоге `SMT/`;
отдельно клонировать ее не нужно.

Проверка окружения:

```powershell
python -m pip check
python -m py_compile app.py SMT\smt_model\modeling_smt.py
```

Не устанавливайте одновременно `opencv-python` и
`opencv-python-headless`: для этого приложения нужен только headless-вариант.