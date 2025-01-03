import random
import numpy as np
from collections import Counter
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from scipy.stats import chi2


# Теоретическая функция распределения
def Fη(p1, p2, x):
    sum = p1
    for k in range(1, x + 1):
        sum += (1 - p1)**k * (1 - p2)**(k - 1) * p2 + (1 - p1)**k * (1 - p2)**k * p1
    return sum


def simulate_experiment(p1, p2, n):
    results = []

    for _ in range(n):
        y = 0
        while True:

            # Первый бросает
            if random.random() < p1:

                break           # Первый попал, игра заканчивается

            # Второй бросает
            y += 1
            if random.random() < p2:
                break           # Второй попал, игра заканчивается

        results.append(y)

    return results



def main():

    p1 = float(input("Введите вероятность попадания первого баскетболиста (p1): "))
    p2 = float(input("Введите вероятность попадания второго баскетболиста (p2): "))
    n = int(input("Введите количество экспериментов (n): "))

    results = simulate_experiment(p1, p2, n)
    results.sort()
    counter = Counter(results)
    yi = sorted(counter.keys())
    ni =[]
    freq = []
    for y in yi:
        ni.append(counter[y]) 

    for i in range(len(yi)):
        freq.append(ni[i]/n)


    table = PrettyTable()
    
    table.field_names = ["Случ. вел-ны (η)", "Кол-во исходов (ni)", "Частоты (ni/n)"]
    for i in range(len(yi)):
        table.add_row([yi[i], ni[i], freq[i]])
    print("\nРезультаты экспериментов (таблица):")
    print(table)



    #-------------------------------------------------------------------- 2 ЧАСТЬ ---------------------------------------------------------------------------------
    
    
    # Теоретические Eη и Dη
    q1 = 1 - p1
    q2 = 1 - p2
    Eη = (1 - p1) / (p1 + p2 - p1 * p2) 
    Dη = (1 - p1) * (1 - p2 + p1 * p2) / ((p1 + p2 - p1 * p2) ** 2)
    
    #Выборочные характеристики
    x_bar = sum(results) / n
    S2 = sum([(xi - x_bar) ** 2 for xi in results]) / n
    R = max(results) - min(results)
    if n % 2 == 0:
        Me_bar = (results[n // 2] + results[n // 2 + 1]) / 2 
    else:
        Me_bar = results[n // 2 + 1]
    Eη_diff = abs(Eη - x_bar)
    Dη_diff = abs(Dη - S2)

    table2 = PrettyTable()
    table2.field_names = ["Eη", "x̄", "|Eη - x̄|", "Dη", "S^2", "|Dη - S^2|", "Mē", "R"]
    table2.add_row([Eη, x_bar, Eη_diff, Dη, S2, Dη_diff, Me_bar, R])
    print(table2)


    # P(η = k) = (1 - p1)^k * (1 - p2)^(k-1) * p2 + (1 - p1)^k * (1 - p2)^(k) * p1 
    # Вычисление теоретических вероятностей и отклонений
    theoretical_probs = {}
    for y in results:
        if y == 0:
            P_y = p1
        else:
            P_y = (1 - p1)**y * (1 - p2)**(y - 1) * p2 + (1 - p1)**y * (1 - p2)**y * p1
        theoretical_probs[y] = P_y

    for y in results:
        list = []
        list.append(abs(freq[yi.index(y)] - theoretical_probs[y]))
    max_diff = max(list)

    table3 = PrettyTable()
    table3.field_names = ["yj", "P({η = yj})", "nj/n"]
    for y in yi:
        table3.add_row([y, theoretical_probs[y], freq[yi.index(y)]])
    print(table3)

    print("Максимальное отклонение между теоретическими и выборочными вероятностями: %.3f" % max_diff)


   
   # Теоретическая и выборочная функции распределения
    x_values = np.arange(0, results[-1] + 1)
    Fη_values = [Fη(p1, p2, x) for x in x_values]
    Fη_b_values = [len([xi for xi in results if xi <= x]) / n for x in x_values]
   
    # Вычисление меры расхождения между теоретической и выборочной функциями распределения
    for j in range(1, n + 1):
        d_values = []
        d_values.append(max(j / n - Fη(p1, p2, results[j - 1]), Fη(p1, p2, results[j - 1]) - (j - 1) / n))
    D = max(d_values)
    # Построение графиков
    print("Рисовать график? y = 1, n = 0")
    if int(input()) == 1:
        plt.step(x_values, Fη_values, where='post', label='Теоретическая Fη(x)')
        plt.step(x_values, Fη_b_values, where='post', label='Выборочная F̂η(x)', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.title('Теор. и выб. ф-ии распределения, D = %.3f' % D)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("График не отображен")
    

    # -------------------------------------------------------------------- 3 ЧАСТЬ ---------------------------------------------------------------------------------


    # Ввод числа интервалов k
    while True:
        k = int(input("Введите число интервалов k для критерия χ²: "))
        if k <= 1:
            print("k должно быть > 1")
            continue
        break
        

    # Ввод границ интервалов: z_1, z_2, ..., z_(k-1)
    # Интервалы: Δ1'' = (-inf, z_1), Δ2'' = [z_1, z_2), ..., Δk'' = (z_[k-1], inf)
    zs = []
    if k > 1:
        print(f"Введите {k - 1} точек разделения интервалов в порядке возрастания:")
    for i in range(k - 1):
        z = float(input(f"z_{i + 1} = "))
        zs.append(z)

    zs.sort() 

    # Функция для определения в какой интервал попадает значение η
    def interval_index(x_val):
        # (-inf, z_1), [z_1, z_2), ..., (z_[k-1], inf)
        for j in range(k - 1):
            if x_val <= zs[j]:
                return j
        return k-1

    # Подсчёт nj
    nj = np.zeros(k)
    for val in results:
        idx = interval_index(val)
        nj[idx] += 1

    # Подсчёт qj
    # qj = P(η c Δj''), вычисляем суммированием вероятностей теоретического распределения
    # Так как η - дискретная с.в., мы просто суммируем те P(y), для y попадающих в интервал.
    def in_interval(y, j):
        # Проверяем, попадает ли y в интервал j
        if j == 0:
            return y <= zs[0] if (k > 1) else True # Если k=1, тогда один интервал -&infin;,&infin;
        elif j == k-1:
            return y > zs[-1] if (k > 1) else True
        else:
            return (y > zs[j - 1]) and (y <= zs[j])

    qj = np.zeros(k)
    for j in range(k):
        qj[j] = sum(theoretical_probs[y] for y in theoretical_probs if in_interval(y, j))

    # Вычисление статистики R_0
    R0 = sum(((nj[j] - n * qj[j])**2) / (n * qj[j]) for j in range(k))

    #гипотеза - распределение не совпадает с тем, что мы вычисляем
    # Ввод уровня значимости α
    while True:
        try:
            alpha = float(input("Введите уровень значимости α, 0 < α < 1: "))
            if alpha <= 0 or alpha >= 1:
                print("α должно быть в (0,1).")
                continue
            break
        except ValueError:
            print("Некорректный ввод")

    # Для χ²-критерия:
    # Число степеней свободы: r = k - 1
    r = k - 1
    
    # p-value = Fχ²_r(R0), где Fχ²_r - функция распределения χ² с r степенями свободы
    p_value = chi2.cdf(R0, r)

    # Вывод результатов проверки гипотезы
    print("\nПроверка гипотезы χ²:")
    print(f"Число интервалов k = {k}")
    print(f"Границы интервалов: {zs}")
    print("qj:", qj)
    print("nj:", nj)
    print(f"Статистика R_0 = {R0:.4f}")
    print(f"p-value = {p_value:.4f}")

    if p_value < alpha:
        print(f"p-value < α, отвергаем гипотезу H_0.")
    else:
        print(f"p-value >= α, нет оснований отвергать гипотезу H_0.")





if __name__ == "__main__":
    main()