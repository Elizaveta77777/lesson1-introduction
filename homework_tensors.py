import torch
import sys


def check_tensor(tensor, expected_shape=None, expected_dtype=None, name=""):
    """
    Вспомогательная функция для проверки тензора.
    
    Args:
        tensor: проверяемый тензор
        expected_shape: ожидаемая форма (опционально)
        expected_dtype: ожидаемый тип данных (опционально)
        name: имя тензора для вывода
    """
    print(f"\n{name}:")
    print(f"  Форма: {tensor.shape}")
    print(f"  Тип данных: {tensor.dtype}")
    print(f"  Устройство: {tensor.device}")
    print(f"  Значения:\n{tensor}")
    
    if expected_shape and tensor.shape != torch.Size(expected_shape):
        print(f"   ОШИБКА: ожидалась форма {expected_shape}, получена {tensor.shape}")
    if expected_dtype and tensor.dtype != expected_dtype:
        print(f"   ОШИБКА: ожидался тип {expected_dtype}, получен {tensor.dtype}")


# ============================================================
# 1.1 Создание тензоров 
# ============================================================

print("=" * 60)
print("ЗАДАНИЕ 1.1: Создание тензоров")
print("=" * 60)

# Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor_3x4 = torch.rand(3, 4)
check_tensor(tensor_3x4, expected_shape=(3, 4), name="Тензор 3x4 (случайные числа)")

# Тензор размером 2x3x4, заполненный нулями
tensor_2x3x4 = torch.zeros(2, 3, 4)
check_tensor(tensor_2x3x4, expected_shape=(2, 3, 4), name="Тензор 2x3x4 (нули)")

# Тензор размером 5x5, заполненный единицами
tensor_5x5 = torch.ones(5, 5)
check_tensor(tensor_5x5, expected_shape=(5, 5), name="Тензор 5x5 (единицы)")

# Тензор размером 4x4 с числами от 0 до 15 (используем reshape)
tensor_4x4 = torch.arange(16).reshape(4, 4)
check_tensor(tensor_4x4, expected_shape=(4, 4), name="Тензор 4x4 (0-15 через reshape)")


# ============================================================
# 1.2 Операции с тензорами 
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1.2: Операции с тензорами")
print("=" * 60)

# Создаём тензоры A (3x4) и B (4x3)
A = torch.rand(3, 4)
B = torch.rand(4, 3)

print(f"\nТензор A (3x4):\n{A}")
print(f"\nТензор B (4x3):\n{B}")

# Транспонирование тензора A
A_transposed = A.t()  # или A.T, или torch.transpose(A, 0, 1)
print(f"\nТранспонированный A (4x3):\n{A_transposed}")
print(f"Форма: {A_transposed.shape}")

# Матричное умножение A и B
# A (3x4) @ B (4x3) = результат (3x3)
matmul_result = torch.matmul(A, B)
print(f"\nМатричное умножение A @ B (3x3):\n{matmul_result}")
print(f"Форма: {matmul_result.shape}")

# Поэлементное умножение A и транспонированного B
# A (3x4) * B.T (3x4) = результат (3x4)
B_transposed = B.t()
elementwise_result = A * B_transposed
print(f"\nПоэлементное умножение A * B.T (3x4):\n{elementwise_result}")
print(f"Форма: {elementwise_result.shape}")

# Сумма всех элементов тензора A
sum_A = torch.sum(A)
print(f"\nСумма всех элементов тензора A: {sum_A.item():.4f}")


# ============================================================
# 1.3 Индексация и срезы 
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1.3: Индексация и срезы")
print("=" * 60)

# Создаём тензор размером 5x5x5
tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)
print(f"\nИсходный тензор 5x5x5:\n{tensor_5x5x5}")

# Первая строка (первый слой, первая строка, все столбцы)
first_row = tensor_5x5x5[0, 0, :]
print(f"\nПервая строка [0, 0, :]:\n{first_row}")
print(f"Форма: {first_row.shape}")

# Последний столбец (все слои, все строки, последний столбец)
last_column = tensor_5x5x5[:, :, -1]
print(f"\nПоследний столбец [:, :, -1]:\n{last_column}")
print(f"Форма: {last_column.shape}")

# Подматрица размером 2x2 из центра тензора
# Центр 5x5x5 — это индексы [1:3, 1:3, 1:3] или [2, 2, :] для одного слоя
center_submatrix = tensor_5x5x5[1:3, 1:3, 2]
print(f"\nПодматрица 2x2 из центра [1:3, 1:3, 2]:\n{center_submatrix}")
print(f"Форма: {center_submatrix.shape}")

# Все элементы с чётными индексами
# Используем срезы с шагом 2
even_indices = tensor_5x5x5[::2, ::2, ::2]
print(f"\nЭлементы с чётными индексами [::2, ::2, ::2]:\n{even_indices}")
print(f"Форма: {even_indices.shape}")


# ============================================================
# 1.4 Работа с формами 
# ============================================================

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1.4: Работа с формами")
print("=" * 60)

# Создаём тензор размером 24 элемента
tensor_24 = torch.arange(24)
print(f"\nИсходный тензор (24 элемента):\n{tensor_24}")
print(f"Форма: {tensor_24.shape}")

# Преобразуем в форму 2x12
tensor_2x12 = tensor_24.reshape(2, 12)
print(f"\nФорма 2x12:\n{tensor_2x12}")

# Преобразуем в форму 3x8
tensor_3x8 = tensor_24.reshape(3, 8)
print(f"\nФорма 3x8:\n{tensor_3x8}")

# Преобразуем в форму 4x6
tensor_4x6 = tensor_24.reshape(4, 6)
print(f"\nФорма 4x6:\n{tensor_4x6}")

# Преобразуем в форму 2x3x4
tensor_2x3x4 = tensor_24.reshape(2, 3, 4)
print(f"\nФорма 2x3x4:\n{tensor_2x3x4}")

# Преобразуем в форму 2x2x2x3
tensor_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
print(f"\nФорма 2x2x2x3:\n{tensor_2x2x2x3}")


# ============================================================
# Тестирование на простых примерах
# ============================================================

print("\n" + "=" * 60)
print("ТЕСТИРОВАНИЕ: Проверка на простых примерах")
print("=" * 60)

# Тест 1: Проверка создания тензоров
assert torch.zeros(2, 3).shape == torch.Size([2, 3]), "Ошибка создания нулевого тензора"
assert torch.ones(3, 3).shape == torch.Size([3, 3]), "Ошибка создания единичного тензора"
assert torch.rand(2, 2).shape == torch.Size([2, 2]), "Ошибка создания случайного тензора"
print(" Тест 1 пройден: создание тензоров работает корректно")

# Тест 2: Проверка матричного умножения
A_test = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B_test = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
result = torch.matmul(A_test, B_test)
assert torch.allclose(result, expected), f"Ошибка matmul: ожидалось {expected}, получено {result}"
print(" Тест 2 пройден: матричное умножение работает корректно")

# Тест 3: Проверка транспонирования
A_test = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
expected_T = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float32)
assert torch.allclose(A_test.t(), expected_T), "Ошибка транспонирования"
print(" Тест 3 пройден: транспонирование работает корректно")

# Тест 4: Проверка reshape
tensor_test = torch.arange(6)
expected_reshape = torch.tensor([[0, 1, 2], [3, 4, 5]])
assert torch.equal(tensor_test.reshape(2, 3), expected_reshape), "Ошибка reshape"
print(" Тест 4 пройден: reshape работает корректно")

# Тест 5: Проверка индексации
tensor_test = torch.arange(27).reshape(3, 3, 3)
assert tensor_test[0, 0, 0].item() == 0, "Ошибка индексации [0,0,0]"
assert tensor_test[-1, -1, -1].item() == 26, "Ошибка индексации [-1,-1,-1]"
print(" Тест 5 пройден: индексация работает корректно")

print("\n" + "=" * 60)
print(" ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
print("=" * 60)