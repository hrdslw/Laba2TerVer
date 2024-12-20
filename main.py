import random
from collections import Counter

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

    counter = Counter(results)
    unique_values = counter.keys()

    yi_row = " ".join(map(str, unique_values))
    ni_row = " ".join(f"{counter[y]}" for y in unique_values)
    frequencies_row = " ".join(f"{counter[eta] / n:.4f}" for eta in unique_values)

    # Вывод результатов
    print("\nРезультаты эксперимента:")
    print("Случайные величины (η):", yi_row)
    print("Количество исходов ni:", ni_row)
    print("Частоты (ni/n): ", frequencies_row)

if __name__ == "__main__":
    main()