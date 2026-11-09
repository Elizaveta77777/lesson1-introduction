import torch
import math


def check_requires_grad(tensor, name=""):
    """
    Проверяет, что тензор имеет requires_grad=True.
    
    Args:
        tensor: проверяемый тензор
        name: имя тензора для вывода
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} должен быть torch.Tensor, получен {type(tensor)}")
    if not tensor.requires_grad:
        raise ValueError(f"{name} должен иметь requires_grad=True")


def check_scalar(tensor, name=""):
    """
    Проверяет, что тензор является скаляром.
    
    Args:
        tensor: проверяемый тензор
        name: имя тензора для вывода
    """
    if tensor.numel() != 1:
        raise ValueError(f"{name} должен быть скаляром, форма {tensor.shape}")


# ============================================================
# 2.1 Простые вычисления с градиентами (8 баллов)
# ============================================================

print("=" * 70)
print("ЗАДАНИЕ 2.1: Простые вычисления с градиентами")
print("=" * 70)

# Создаём тензоры x, y, z с requires_grad=True
# requires_grad=True говорит PyTorch отслеживать все операции с этим тензором
# и автоматически вычислять градиенты при вызове .backward()
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# Проверки
check_requires_grad(x, "x")
check_requires_grad(y, "y")
check_requires_grad(z, "z")
check_scalar(x, "x")
check_scalar(y, "y")
check_scalar(z, "z")

print(f"\nИсходные значения:")
print(f"  x = {x.item()}, y = {y.item()}, z = {z.item()}")
print(f"  Тип данных: {x.dtype}")
print(f"  Устройство: {x.device}")

# Вычисляем функцию: f(x,y,z) = x² + y² + z² + 2*x*y*z
# PyTorch автоматически строит граф вычислений
f = x**2 + y**2 + z**2 + 2 * x * y * z

print(f"\nФункция: f(x,y,z) = x² + y² + z² + 2*x*y*z")
print(f"  f({x.item()}, {y.item()}, {z.item()}) = {f.item()}")

# Вызываем backward() для вычисления градиентов
# По умолчанию backward() вычисляет градиент скалярной функции
f.backward()

# Градиенты сохраняются в атрибуте .grad каждого тензора
print(f"\nГрадиенты (после f.backward()):")
print(f"  df/dx = {x.grad.item()}")
print(f"  df/dy = {y.grad.item()}")
print(f"  df/dz = {z.grad.item()}")

# Аналитическая проверка:
# f = x² + y² + z² + 2xyz
# df/dx = 2x + 2yz
# df/dy = 2y + 2xz
# df/dz = 2z + 2xy

x_val, y_val, z_val = 2.0, 3.0, 1.0
df_dx_analytical = 2 * x_val + 2 * y_val * z_val
df_dy_analytical = 2 * y_val + 2 * x_val * z_val
df_dz_analytical = 2 * z_val + 2 * x_val * y_val

print(f"\nАналитическая проверка:")
print(f"  df/dx = 2x + 2yz = 2*{x_val} + 2*{y_val}*{z_val} = {df_dx_analytical}")
print(f"  df/dy = 2y + 2xz = 2*{y_val} + 2*{x_val}*{z_val} = {df_dy_analytical}")
print(f"  df/dz = 2z + 2xy = 2*{z_val} + 2*{x_val}*{y_val} = {df_dz_analytical}")

# Проверяем совпадение
assert math.isclose(x.grad.item(), df_dx_analytical, abs_tol=1e-5), "Ошибка в df/dx"
assert math.isclose(y.grad.item(), df_dy_analytical, abs_tol=1e-5), "Ошибка в df/dy"
assert math.isclose(z.grad.item(), df_dz_analytical, abs_tol=1e-5), "Ошибка в df/dz"
print("\n✅ Градиенты совпадают с аналитическими!")


# ============================================================
# 2.2 Градиент функции потерь (9 баллов)
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2.2: Градиент функции потерь (MSE)")
print("=" * 70)


def compute_mse(y_pred, y_true):
    """
    Вычисляет Mean Squared Error (MSE).
    
    Args:
        y_pred: предсказанные значения (torch.Tensor)
        y_true: истинные значения (torch.Tensor)
    
    Returns:
        MSE как скалярный тензор
    
    Raises:
        ValueError: если формы тензоров не совпадают
        TypeError: если входные данные не тензоры
    """
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("y_pred и y_true должны быть torch.Tensor")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Формы не совпадают: y_pred {y_pred.shape} vs y_true {y_true.shape}")
    n = y_pred.numel()
    return torch.sum((y_pred - y_true) ** 2) / n


# Создаём данные
# x — входные данные (features)
# y_true — истинные значения (targets)
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])  # идеальная линейная зависимость y = 2x

# Параметры модели w и b с requires_grad=True
# Инициализируем случайными значениями
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

print(f"\nИсходные данные:")
print(f"  x = {x_data.tolist()}")
print(f"  y_true = {y_true.tolist()}")
print(f"  Начальные параметры: w = {w.item()}, b = {b.item()}")

# Forward pass: вычисляем предсказания
# y_pred = w * x + b
y_pred = w * x_data + b
print(f"\nПредсказания (y_pred = w*x + b): {y_pred.tolist()}")

# Вычисляем MSE через нашу функцию
mse = compute_mse(y_pred, y_true)
print(f"\nMSE = {mse.item():.4f}")

# Вычисляем градиенты
mse.backward()

print(f"\nГрадиенты MSE:")
print(f"  dMSE/dw = {w.grad.item():.4f}")
print(f"  dMSE/db = {b.grad.item():.4f}")

# Аналитическая проверка:
# y_pred = w*x + b
# MSE = (1/n) * Σ(w*x + b - y_true)²
# dMSE/dw = (2/n) * Σ(w*x + b - y_true) * x
# dMSE/db = (2/n) * Σ(w*x + b - y_true)

# Используем detach(), чтобы получить значения без градиентов
w_val = w.detach()
b_val = b.detach()
errors = (w_val * x_data + b_val - y_true)
dMSE_dw_analytical = (2.0 / len(x_data)) * torch.sum(errors * x_data).item()
dMSE_db_analytical = (2.0 / len(x_data)) * torch.sum(errors).item()

print(f"\nАналитическая проверка:")
print(f"  dMSE/dw = (2/n) * Σ(error * x) = {dMSE_dw_analytical:.4f}")
print(f"  dMSE/db = (2/n) * Σ(error) = {dMSE_db_analytical:.4f}")

assert math.isclose(w.grad.item(), dMSE_dw_analytical, abs_tol=1e-4), "Ошибка в dMSE/dw"
assert math.isclose(b.grad.item(), dMSE_db_analytical, abs_tol=1e-4), "Ошибка в dMSE/db"
print("\n✅ Градиенты MSE совпадают с аналитическими!")

# Демонстрация: один шаг градиентного спуска
learning_rate = 0.01
w_new = w.item() - learning_rate * w.grad.item()
b_new = b.item() - learning_rate * b.grad.item()
print(f"\nОдин шаг градиентного спуска (lr={learning_rate}):")
print(f"  w_old = {w.item():.4f} → w_new = {w_new:.4f}")
print(f"  b_old = {b.item():.4f} → b_new = {b_new:.4f}")


# ============================================================
# 2.3 Цепное правило (8 баллов)
# ============================================================

print("\n" + "=" * 70)
print("ЗАДАНИЕ 2.3: Цепное правило")
print("=" * 70)

# Реализуем составную функцию: f(x) = sin(x² + 1)
# Найдём градиент df/dx

# Создаём тензор x
x_comp = torch.tensor(1.5, requires_grad=True)
check_requires_grad(x_comp, "x_comp")
print(f"\nИсходное значение: x = {x_comp.item()}")

# Вычисляем f(x) = sin(x² + 1)
# PyTorch автоматически применяет цепное правило:
# df/dx = cos(x² + 1) * 2x
f_comp = torch.sin(x_comp ** 2 + 1)
print(f"\nФункция: f(x) = sin(x² + 1)")
print(f"  f({x_comp.item()}) = sin({x_comp.item()}² + 1) = sin({x_comp.item()**2 + 1}) = {f_comp.item():.4f}")

# Способ 1: через backward()
# retain_graph=True сохраняет граф для повторного использования
f_comp.backward(retain_graph=True)
grad_backward = x_comp.grad.item()
print(f"\nСпособ 1 (через backward() с retain_graph=True):")
print(f"  df/dx = {grad_backward:.4f}")

# Сбрасываем градиент для следующего вычисления
x_comp.grad = None

# Способ 2: через torch.autograd.grad (не изменяет .grad атрибут)
# Теперь граф ещё доступен благодаря retain_graph=True
grad_autograd = torch.autograd.grad(f_comp, x_comp, create_graph=False)[0].item()
print(f"\nСпособ 2 (через torch.autograd.grad):")
print(f"  df/dx = {grad_autograd:.4f}")

# Аналитическая проверка:
# f(x) = sin(x² + 1)
# df/dx = cos(x² + 1) * 2x (цепное правило: производная sin(u) * производная u)
x_val = 1.5
df_dx_analytical = math.cos(x_val ** 2 + 1) * 2 * x_val
print(f"\nАналитическая проверка (цепное правило):")
print(f"  df/dx = cos(x² + 1) * 2x = cos({x_val**2 + 1}) * 2*{x_val} = {df_dx_analytical:.4f}")

assert math.isclose(grad_backward, df_dx_analytical, abs_tol=1e-5), "Ошибка в backward()"
assert math.isclose(grad_autograd, df_dx_analytical, abs_tol=1e-5), "Ошибка в autograd.grad"
print("\n✅ Оба способа дают одинаковый результат, совпадающий с аналитическим!")

# Демонстрация цепного правила на более сложной функции
print("\n" + "-" * 70)
print("Дополнительный пример: цепное правило для f(x) = exp(sin(x))")
print("-" * 70)

x_chain = torch.tensor(2.0, requires_grad=True)
f_chain = torch.exp(torch.sin(x_chain))
f_chain.backward()

# Аналитически: df/dx = exp(sin(x)) * cos(x)
analytical_chain = math.exp(math.sin(2.0)) * math.cos(2.0)
print(f"  x = {x_chain.item()}")
print(f"  f(x) = exp(sin(x)) = {f_chain.item():.4f}")
print(f"  df/dx (PyTorch) = {x_chain.grad.item():.4f}")
print(f"  df/dx (аналитически) = exp(sin(x)) * cos(x) = {analytical_chain:.4f}")
assert math.isclose(x_chain.grad.item(), analytical_chain, abs_tol=1e-5)
print("  ✅ Совпадает!")


# ============================================================
# Дополнительные тесты
# ============================================================

print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ: Проверка autograd на простых примерах")
print("=" * 70)


def test_gradient_simple():
    """Тест 1: Градиент простой функции f(x) = x²"""
    x_test = torch.tensor(3.0, requires_grad=True)
    f_test = x_test ** 2
    f_test.backward()
    assert math.isclose(x_test.grad.item(), 6.0, abs_tol=1e-5), "Ошибка: d(x²)/dx = 2x = 6"
    return True


def test_gradient_linear():
    """Тест 2: Градиент линейной функции f(x) = 3x + 2"""
    x_test = torch.tensor(5.0, requires_grad=True)
    f_test = 3 * x_test + 2
    f_test.backward()
    assert math.isclose(x_test.grad.item(), 3.0, abs_tol=1e-5), "Ошибка: d(3x+2)/dx = 3"
    return True


def test_gradient_sum():
    """Тест 3: Градиент суммы f(x,y) = x + y"""
    x_test = torch.tensor(2.0, requires_grad=True)
    y_test = torch.tensor(3.0, requires_grad=True)
    f_test = x_test + y_test
    f_test.backward()
    assert math.isclose(x_test.grad.item(), 1.0, abs_tol=1e-5), "Ошибка: d(x+y)/dx = 1"
    assert math.isclose(y_test.grad.item(), 1.0, abs_tol=1e-5), "Ошибка: d(x+y)/dy = 1"
    return True


def test_gradient_product():
    """Тест 4: Градиент произведения f(x,y) = x*y"""
    x_test = torch.tensor(2.0, requires_grad=True)
    y_test = torch.tensor(3.0, requires_grad=True)
    f_test = x_test * y_test
    f_test.backward()
    assert math.isclose(x_test.grad.item(), 3.0, abs_tol=1e-5), "Ошибка: d(xy)/dx = y = 3"
    assert math.isclose(y_test.grad.item(), 2.0, abs_tol=1e-5), "Ошибка: d(xy)/dy = x = 2"
    return True


def test_requires_grad_flag():
    """Тест 5: Проверка requires_grad"""
    x_no_grad = torch.tensor(5.0)
    assert not x_no_grad.requires_grad, "Ошибка: requires_grad должен быть False по умолчанию"
    x_grad = torch.tensor(5.0, requires_grad=True)
    assert x_grad.requires_grad, "Ошибка: requires_grad должен быть True"
    return True


def test_autograd_grad():
    """Тест 6: Проверка torch.autograd.grad"""
    x_test = torch.tensor(2.0, requires_grad=True)
    f_test = x_test ** 3
    grad = torch.autograd.grad(f_test, x_test)[0].item()
    assert math.isclose(grad, 12.0, abs_tol=1e-5), "Ошибка: d(x³)/dx = 3x² = 12"
    return True


def test_mse_function():
    """Тест 7: Проверка функции compute_mse"""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.0, 2.0, 3.0])
    mse = compute_mse(y_pred, y_true)
    assert math.isclose(mse.item(), 0.0, abs_tol=1e-5), "MSE для одинаковых тензоров должен быть 0"
    
    y_pred2 = torch.tensor([2.0, 2.0, 2.0])
    y_true2 = torch.tensor([1.0, 1.0, 1.0])
    mse2 = compute_mse(y_pred2, y_true2)
    assert math.isclose(mse2.item(), 1.0, abs_tol=1e-5), "MSE должен быть 1.0"
    return True


def test_mse_shape_check():
    """Тест 8: Проверка обработки ошибок в compute_mse"""
    try:
        compute_mse(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))
        return False  # Должно было выбросить ошибку
    except ValueError:
        return True


# Запуск всех тестов
tests = [
    ("Градиент x² = 2x", test_gradient_simple),
    ("Градиент 3x+2 = 3", test_gradient_linear),
    ("Градиенты x+y", test_gradient_sum),
    ("Градиенты x*y", test_gradient_product),
    ("requires_grad флаг", test_requires_grad_flag),
    ("torch.autograd.grad", test_autograd_grad),
    ("Функция MSE", test_mse_function),
    ("Проверка форм в MSE", test_mse_shape_check),
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
    print("✅ ВСЕ ТЕСТЫ АВТОГРАДА ПРОЙДЕНЫ!")
    print("=" * 70)
else:
    print(f"\n⚠️ Не все тесты пройдены ({passed}/{len(tests)})")

# Итоговая сводка
print("\n" + "=" * 70)
print("КЛЮЧЕВЫЕ КОНЦЕПЦИИ AUTOGRAD:")
print("=" * 70)
print("""
1. requires_grad=True — включает отслеживание операций с тензором
2. .backward() — вычисляет градиенты и сохраняет их в .grad
3. retain_graph=True — сохраняет граф для повторного использования
4. torch.autograd.grad() — вычисляет градиенты без сохранения в .grad
5. Цепное правило — автоматически применяется PyTorch для составных функций
6. Градиентный спуск: w_new = w_old - lr * gradient
7. MSE = (1/n) * Σ(y_pred - y_true)² — стандартная функция потерь для регрессии
""")

# Информация об устройстве
print(f"Устройство тензоров: {x.device}")
print(f"Версия PyTorch: {torch.__version__}")