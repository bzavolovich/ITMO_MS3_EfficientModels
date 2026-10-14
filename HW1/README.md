# HW1

## Окружение

Google Colab, Tesla T4, FP32. PyTorch `2.11.0+cu130`, CUDA `13.0`, cuDNN `9.27.0`.

## Воспроизведение

Открыть [HW1_Colab.ipynb](HW1_Colab.ipynb) в Colab, выбрать GPU T4 и выполнить все ячейки.

Пересчитать параметры и графики по сохранённым измерениям (из корня репозитория):

```bash
python3 -m pip install numpy pandas scipy matplotlib
python3 HW1/calibrate.py
```

## Результаты

- 132 конфигурации: 63 для калибровки, 69 для валидации; OOM в основной сетке не получилось.
- При `B=868` — OOM.

Измерения, параметры и графики лежат в [results/](results/).
