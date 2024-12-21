import random
import numpy as np
from collections import Counter
from prettytable import PrettyTable
from matplotlib import pyplot as plt


# Теоретическая функция распределения
def Fη(p1, p2, x):
    sum = p1
    for k in range(1, x + 1):
        sum += (1 - p1)**k * (1 - p2)**(k - 1) * p2
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
    Eη = (1 - p1) / (p1 + p2 - p1 * p2)
    Dη = (1 - p1) * (1 - p2 + p1 * p2) / ((p1 + p2 - p1 * p2) ** 2)
    
    #Выборочные характеристики
    x_bar = sum(results) / n
    S2 = sum([(xi - x_bar) ** 2 for xi in results]) / n
    R = max(results) - min(results)
    if n % 2 == 0:
        Me_bar = (results[n // 2] + results[n // 2 + 1]) / 2 #TODO проверить правильность формулы для четного n 
    else:
        Me_bar = results[n // 2 + 1]
    Eη_diff = abs(Eη - x_bar)
    Dη_diff = abs(Dη - S2)

    table2 = PrettyTable()
    table2.field_names = ["Eη", "x̄", "|Eη - x̄|", "Dη", "S^2", "|Dη - S^2|", "Mē", "R"]
    table2.add_row([Eη, x_bar, Eη_diff, Dη, S2, Dη_diff, Me_bar, R])
    print(table2)



    # Вычисление теоретических вероятностей и отклонений
    theoretical_probs = {}
    for y in results:
        if y == 0:
            P_y = p1
        else:
            P_y = (1 - p1)**y * (1 - p2)**(y - 1) * p2
        theoretical_probs[y] = P_y

    for y in results:
        list = []
        list.append(abs(freq[yi.index(y)] - theoretical_probs[y]))
    max_diff = max(list)
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
    plt.step(x_values, Fη_values, where='post', label='Теоретическая Fη(x)')
    plt.step(x_values, Fη_b_values, where='post', label='Выборочная F̂η(x)', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Функции распределения, D = %.3f' % D)
    plt.legend()
    plt.grid(True)
    plt.show()

    

    # -------------------------------------------------------------------- 3 ЧАСТЬ ---------------------------------------------------------------------------------
    




if __name__ == "__main__":
    main()