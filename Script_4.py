import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Параметры для построения первого графика
B = 1e-18  # Пример значения B
b = 2  # Пример значения b
Eb = -38.7  # Пример значения E_b
gamma = 0.25


# Функция для вычисления первого выражения (старое)
def v1_old(u):
    return gamma * B * np.abs(u) * np.exp((np.abs(b) * np.sqrt(np.abs(u)) - Eb))


# Параметры для построения второго графика
epsilon = 0.001  # Значение epsilon
alpha = 1.5  # Значение alpha
eta = 0.28  # Значение eta


# Функция для вычисления второго выражения
def v2(u):
    return (alpha * u) - eta


# Параметры для нового графика
p0 =  1e2  # Значение p0


# Функция для вычисления нового выражения
def v1_new(u):
    return gamma * (u / p0)


# Диапазон значений u
u_vals = np.linspace(0, 3, 5000)

# Вычисление значений для обоих выражений
v_vals_1_old = v1_old(u_vals)
v_vals_2 = v2(u_vals)
v_vals_1_new = v1_new(u_vals)


# Функция для поиска пересечений
def find_intersection(func1, func2, u_vals):
    # Функция для нахождения пересечений
    def equation(u):
        return func1(u) - func2(u)

    # Ищем пересечения методом fsolve
    intersections = []
    for i in range(len(u_vals) - 1):
        u_start = u_vals[i]
        u_end = u_vals[i + 1]

        # Если значения функций имеют разные знаки на отрезке, то ищем пересечение
        if equation(u_start) * equation(u_end) < 0:
            u_intersection = fsolve(equation, (u_start + u_end) / 2)
            intersections.append(u_intersection[0])

    return np.array(intersections)


# Находим точки пересечений для v2(u) с v1_old(u) и v1_new(u)
intersections_v2_v1old = find_intersection(v2, v1_old, u_vals)
intersections_v2_v1new = find_intersection(v2, v1_new, u_vals)

# Построение графиков
plt.figure(figsize=(10, 6))

# Первый график: v1_old(u) = B * |u| * e^(b * sqrt(|u|) - E_b)
plt.plot(u_vals, v_vals_1_old, label=r'$v(u) = B \cdot |u| \cdot e^{{b \cdot \sqrt{|u|} - E_b}}, x=0,$', color='purple')

# Второй график: v2(u) = epsilon * (alpha * u - eta)
plt.plot(u_vals, v_vals_2, label=r'$v_1(u) = \alpha u - \eta$', color='green', linestyle='--')

# Новый график: v1_new(u) = u / p0
plt.plot(u_vals, v_vals_1_new, label=r'$v(u) = \frac{u}{p_0}, x=1 $', color='blue', linestyle='-')

# Отображаем точки пересечений для v2(u) с v1_old(u) и v1_new(u)
plt.scatter(intersections_v2_v1old, v2(intersections_v2_v1old), color='red', zorder=5,
            label=r'Пересечение $v_2(u) = v_1(u)$')
plt.scatter(intersections_v2_v1new, v2(intersections_v2_v1new), color='orange', zorder=5,
            label=r'Пересечение $v_2(u) = v_1^{new}(u)$')

# Выводим координаты точек пересечений
print("Координаты точек пересечений:")
print(f"\nКоличество пересечений v2(u) с v1(u): {len(intersections_v2_v1old)}")
for u in intersections_v2_v1old:
    print(f"u = {u:.4f}, v = {v2(u):.4e}")

print(f"\nКоличество пересечений v2(u) с v1_new(u): {len(intersections_v2_v1new)}")
for u in intersections_v2_v1new:
    print(f"u = {u:.4f}, v = {v2(u):.4e}")

# Настройки графика
plt.xlabel(r'$u$', fontsize=20)
plt.ylabel(r'$v$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()