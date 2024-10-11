from tabulate import tabulate
from numpy import hstack, array, zeros
from re import split


def get_dio():
    dio = input('Введіть своє діофантове рівняння у форматі 15x+9y=27: ')
    print('\nВаше діофантове рівняння: ', dio)
    dio_list = split(r'[+=]', dio)
    dio_list[0], dio_list[1] = dio_list[0][:-1], dio_list[1][:-1]
    dio_list = [int(x) for x in dio_list]
    return dio_list[0], dio_list[1], dio_list[2]

def euclidian(a, b):
    qi_list = []
    rim1, ri = a, b
    itr = 0

    while True:
        itr += 1
        qi = rim1//ri
        rip1 = rim1 - ri*qi
        print(f'{rim1} = {ri} × {qi} + {rip1}')
        qi_list.append(qi)
        rim1, ri = ri, rip1

        if rip1 == 0:
            return itr, qi_list, rim1


def is_solutions(a, b, c, gcd):
    if c % gcd == 0:
        print(f'{c} ⋮ {gcd} ⇒ розв\'язки є')
        return True
    else:
        return False


def extended_euclidian(itr, qi_list, s1, s2):
    data = zeros((itr, 3))
    data[0][0], data[0][1] = s1, s2

    for i in range(itr):
        data[i][2] = data[i][0] - qi_list[i] * data[i][1]

        if i <= itr-2:
            data[i + 1][0] = data[i][1]
            data[i + 1][1] = data[i][2]
    return data[-1][1], data


def find_coef(a, b, c, gcd, u, v):
    a0, b0, c0 = a//gcd, b//gcd, c//gcd
    x0, y0 = u*c0, v*c0

    return a0, b0, c0, x0, y0


a, b, c = get_dio()

print('1) Алгоритм Евкліда:')
itr, qi_list, gcd = euclidian(a, b)
print(f'd = gcd({a}, {b}) = {gcd}')

print('2) Перевірка на наявність розв\'язків:')
assert is_solutions(a, b, c, gcd), 'розв\'язків нема'

print('3) Знаходження u та v:')
u, data_u = extended_euclidian(itr, qi_list, 1, 0)
v, data_v = extended_euclidian(itr, qi_list, 0, 1)

data = hstack((array(list(range(1, itr+1))).reshape(-1, 1), array(qi_list).reshape(-1, 1), data_u, data_v))
print(tabulate(data, headers=['i', 'qi', 'ui-1', 'ui', 'ui+1', 'vi-1', 'vi', 'vi+1'], tablefmt='orgtbl'))
print(f'⇒ u = {int(u)}, v = {int(v)}')

print('4) Знаходження коефіцієнтів, потрібних для системи:')
a0, b0, c0, x0, y0 = find_coef(a, b, c, gcd, u, v)
a0, b0, c0, x0, y0 = map(int, (a0, b0, c0, x0, y0))
print(f'a0 = a/d = {a0}, b0 = b/d = {b0}, c0 = c/d = {c0}')
print(f'x0 = u * c0 = {x0}, y0 = v * c0 = {y0}')

print('5) Система розв\'язків')
print(f'|x = {x0} + {b0} * t|')
print(f'|y = {y0} - {a0} * t|')
