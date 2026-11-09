"""
Домашнее задание к уроку 1: Основы PyTorch
Задание 3: Сравнение производительности CPU vs CUDA (20 баллов)
"""

import torch
import time


def measure_time_cpu(func, num_runs=10):
    """Измеряет время выполнения функции на CPU."""
    times = []
    for _ in range(num_runs):
        start = time.time()
        func()
        end = time.time()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def measure_time_gpu(func, num_runs=10):
    """Измеряет время выполнения функции на GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_runs):
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return sum(times) / len(times)


# 3.1 Подготовка данных
print("=" * 70)
print("ЗАДАНИЕ 3.1: Подготовка данных")
print("=" * 70)

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

print(f"\nДоступные устройства:")
print(f"  CPU: всегда доступен")
print(f"  GPU (CUDA): {'доступен' if device_gpu else 'НЕДОСТУПЕН'}")

if device_gpu:
    print(f"  Модель GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Версия CUDA: {torch.version.cuda}")

sizes = {
    "64x1024x1024": (64, 1024, 1024),
    "128x512x512": (128, 512, 512),
    "256x256x256": (256, 256, 256),
}

matrices_cpu = {}
matrices_gpu = {}

print(f"\nСоздание больших матриц...")
for name, shape in sizes.items():
    matrix_cpu = torch.randn(*shape, dtype=torch.float32)
    matrices_cpu[name] = matrix_cpu
    print(f"  {name}: форма {shape}, элементов {matrix_cpu.numel():,}")
    if device_gpu:
        try:
            matrix_gpu = matrix_cpu.to(device_gpu)
            matrices_gpu[name] = matrix_gpu
        except RuntimeError as e:
            print(f"  ⚠️ Не удалось перенести {name} на GPU: {e}")

print(f"\n✅ Матрицы созданы")


# 3.2 Функция измерения времени
print("\n" + "=" * 70)
print("ЗАДАНИЕ 3.2: Функция измерения времени")
print("=" * 70)
print(f"\nФункции измерения времени созданы:")
print(f"  measure_time_cpu() — использует time.time()")
print(f"  measure_time_gpu() — использует torch.cuda.Event()")


# 3.3 Сравнение операций
print("\n" + "=" * 70)
print("ЗАДАНИЕ 3.3: Сравнение операций")
print("=" * 70)


def matmul_operation(matrix):
    if matrix.dim() == 3:
        return torch.matmul(matrix, matrix.transpose(-2, -1))
    return torch.matmul(matrix, matrix.T)


def addition_operation(matrix):
    return matrix + matrix


def multiplication_operation(matrix):
    return matrix * matrix


def transpose_operation(matrix):
    if matrix.dim() >= 2:
        return matrix.transpose(-2, -1)
    return matrix


def sum_operation(matrix):
    return torch.sum(matrix)


operations = {
    "Матричное умножение": matmul_operation,
    "Поэлементное сложение": addition_operation,
    "Поэлементное умножение": multiplication_operation,
    "Транспонирование": transpose_operation,
    "Сумма элементов": sum_operation,
}

results = {}

print(f"\n{'='*70}")
print(f"Сравнение производительности")
print(f"{'='*70}")

for size_name, matrix_cpu in matrices_cpu.items():
    print(f"\nРазмер: {size_name} {sizes[size_name]}")
    print(f"{'-'*70}")
    results[size_name] = {}
    for op_name, op_func in operations.items():
        print(f"\n  Операция: {op_name}")
        try:
            cpu_time = measure_time_cpu(lambda: op_func(matrix_cpu), num_runs=5)
            print(f"    CPU: {cpu_time:.2f} мс")
            results[size_name][op_name] = {"cpu": cpu_time}
        except Exception as e:
            print(f"    CPU: ошибка — {e}")
            results[size_name][op_name] = {"cpu": None}
        if device_gpu and size_name in matrices_gpu:
            matrix_gpu = matrices_gpu[size_name]
            try:
                gpu_time = measure_time_gpu(lambda: op_func(matrix_gpu), num_runs=5)
                print(f"    GPU: {gpu_time:.2f} мс")
                results[size_name][op_name]["gpu"] = gpu_time
                if results[size_name][op_name]["cpu"] and gpu_time > 0:
                    speedup = results[size_name][op_name]["cpu"] / gpu_time
                    print(f"    Ускорение: {speedup:.1f}x")
                    results[size_name][op_name]["speedup"] = speedup
            except Exception as e:
                print(f"    GPU: ошибка — {e}")
                results[size_name][op_name]["gpu"] = None
        else:
            results[size_name][op_name]["gpu"] = None

print(f"\n{'='*70}")
print(f"СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print(f"{'='*70}")

for size_name in sizes.keys():
    print(f"\nРазмер: {size_name}")
    print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for op_name in operations.keys():
        cpu_time = results[size_name][op_name].get("cpu")
        gpu_time = results[size_name][op_name].get("gpu")
        speedup = results[size_name][op_name].get("speedup")
        cpu_str = f"{cpu_time:.2f}" if cpu_time else "N/A"
        gpu_str = f"{gpu_time:.2f}" if gpu_time else "N/A"
        speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
        print(f"{op_name:<25} | {cpu_str:<10} | {gpu_str:<10} | {speedup_str:<10}")


# 3.4 Анализ результатов
print("\n" + "=" * 70)
print("ЗАДАНИЕ 3.4: Анализ результатов")
print("=" * 70)

if device_gpu:
    print("""
Анализ результатов (GPU доступен):

1. Какие операции получают наибольшее ускорение на GPU?
   - Матричное умножение (matmul) получает наибольшее ускорение,
     так как GPU оптимизирован для параллельных матричных операций.
   - Поэлементные операции также ускоряются, но меньше.

2. Почему некоторые операции могут быть медленнее на GPU?
   - Транспонирование — простая операция, накладные расходы на передачу
     данных CPU↔GPU могут превышать выигрыш от параллелизма.
   - Для маленьких матриц передача данных занимает больше времени.

3. Как размер матриц влияет на ускорение?
   - Чем больше матрицы, тем больше ускорение на GPU.
   - На маленьких матрицах CPU может быть быстрее.

4. Что происходит при передаче данных между CPU и GPU?
   - Данные копируются из RAM в VRAM.
   - Это занимает время (несколько миллисекунд).
   - Нужно минимизировать передачи: делать вычисления на GPU,
     передавать только результат.
""")
else:
    print("""
Анализ результатов (GPU НЕДОСТУПЕН):

Поскольку GPU недоступен, проведено сравнение разных размеров матриц на CPU.

1. Как размер матриц влияет на время выполнения?
   - Время выполнения растёт с увеличением размера матриц.
   - Матричное умножение имеет сложность O(n³), время растёт кубически.
   - Поэлементные операции имеют сложность O(n), время растёт линейно.

2. Какие операции самые медленные?
   - Матричное умножение — самая ресурсоёмкая операция.
   - Сумма всех элементов требует обхода всего тензора.

3. Какие операции самые быстрые?
   - Транспонирование — не требует вычислений, только изменение strides.
   - Поэлементные операции хорошо векторизуются на CPU.

4. Что происходит при передаче данных между CPU и GPU?
   - В данном эксперименте передача не требуется (всё на CPU).
   - В реальных задачах с GPU нужно минимизировать передачи данных.

5. Практические рекомендации:
   - Для маленьких матриц (< 1000x1000) CPU может быть достаточно.
   - Для больших матриц и нейронных сетей GPU даёт значительное ускорение.
   - Важно держать данные на GPU во время обучения.
""")


# Тестирование
print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ: Проверка на простых примерах")
print("=" * 70)


def test_small_matmul():
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    C = torch.matmul(A, B)
    expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
    assert torch.allclose(C, expected), "Ошибка matmul"
    return True


def test_small_addition():
    A = torch.tensor([1, 2, 3], dtype=torch.float32)
    B = torch.tensor([4, 5, 6], dtype=torch.float32)
    C = A + B
    expected = torch.tensor([5, 7, 9], dtype=torch.float32)
    assert torch.equal(C, expected), "Ошибка сложения"
    return True


def test_small_transpose():
    A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    A_T = A.T
    expected = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float32)
    assert torch.equal(A_T, expected), "Ошибка транспонирования"
    return True


def test_small_sum():
    A = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    s = torch.sum(A)
    assert abs(s.item() - 15.0) < 1e-5, "Ошибка суммы"
    return True


def test_measure_time_cpu():
    def heavier_func():
        A = torch.randn(1000, 1000)
        B = torch.randn(1000, 1000)
        return torch.matmul(A, B)
    t = measure_time_cpu(heavier_func, num_runs=10)
    assert t >= 0, "Время должно быть неотрицательным"
    return True


def test_gpu_fallback():
    matrix = torch.randn(10, 10)
    result = matmul_operation(matrix)
    assert result.shape == (10, 10), f"Ошибка формы: {result.shape}"
    return True


tests = [
    ("Матричное умножение (2x2)", test_small_matmul),
    ("Поэлементное сложение", test_small_addition),
    ("Транспонирование", test_small_transpose),
    ("Сумма элементов", test_small_sum),
    ("Измерение времени CPU", test_measure_time_cpu),
    ("Адаптивность (без GPU)", test_gpu_fallback),
]

passed = 0
for name, test_func in tests:
    try:
        if test_func():
            print(f"✅ Тест пройден: {name}")
            passed += 1
    except Exception as e:
        print(f"❌ Тест провален: {name} — {e}")

print(f"\nПройдено тестов: {passed}/{len(tests)}")

if passed == len(tests):
    print("\n" + "=" * 70)
    print("✅ ВСЕ ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ ПРОЙДЕНЫ!")
    print("=" * 70)
else:
    print(f"\n⚠️ Не все тесты пройдены ({passed}/{len(tests)})")

print("\n" + "=" * 70)
print("ИНФОРМАЦИЯ О СИСТЕМЕ:")
print("=" * 70)
print(f"  Версия PyTorch: {torch.__version__}")
print(f"  CPU: доступен")
print(f"  GPU (CUDA): {'доступен' if device_gpu else 'НЕДОСТУПЕН'}")
if device_gpu:
    print(f"  Модель GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Версия CUDA: {torch.version.cuda}")

print("\n" + "=" * 70)
print("ЗАДАНИЕ 3 ВЫПОЛНЕНО!")
print("=" * 70)