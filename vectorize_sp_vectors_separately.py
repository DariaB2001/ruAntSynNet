from gensim.models import Word2Vec
import csv
from tqdm import tqdm
import json
import numpy as np
import random

# Загрузим word2vec-модели для каждого из временных периодов.
print('Загрузка моделей word2vec...')
w2v_models = 'YOUR_PATH'  # путь к папке, где лежат модели word2vec
pre_soviet_path = f'{w2v_models}/w2v_presoviet/w2v_presoviet.model'
soviet_path = f'{w2v_models}/w2v_soviet/w2v_soviet.model'
post_soviet_path = '{w2v_models}/w2v_postsoviet/w2v_postsoviet.model'
model_pre_soviet = Word2Vec.load(pre_soviet_path)
model_soviet = Word2Vec.load(soviet_path)
model_post_soviet = Word2Vec.load(post_soviet_path)

# Загрузим словари с векторами для pos-тегов и тегов синтаксических связей
print('Загрузка словарей с pos-тегами и тегами синтаксических связей...')
pos_dict = 'YOUR_PATH'
# pos-теги
with open(pos_dict) as f1:
    pos_tags = dict(json.load(f1))
pos_dimension = len(pos_tags[list(pos_tags.keys())[0]])  # размерность векторов pos-тегов
# dep-теги
dep_dict = 'YOUR_PATH'
with open(dep_dict) as f2:
    dep_tags = dict(json.load(f2))
dep_dimension = len(dep_tags[list(dep_tags.keys())[0]])  # размерность векторов dep-тегов


def vectorize_simple_paths(simple_paths_filepath, json_filepath, folder_with_numpy_arrays):
    """
    Функция для векторизации простых путей.
    Получает на вход csv-файл с простыми путями, который содержит строки вида:
    word_pair_id | target1 | target2 | sent_id | simple_path | relation_label
    Вектор каждого векторизованного простого пути записывается в файл .npy.
    Название файла имеет вид S_123 / A_123, что позволяет определить тип отношений.
    Все npy-файлы лежат в папке folder_with_numpy_arrays.
    """

    # Загрузим простые пути из входного файла
    simple_paths = []
    with open(simple_paths_filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            simple_paths.append(row)

    words_ids_filenames = {}  # словарь вида {filename: [word_pair_id, target1, target2]}
    counter = 1  # счётчик простых путей - для формирования названия файлов с numpy-массивами
    for sp in tqdm(simple_paths):  # итеративно проходим по всех простым путям
        word_pair_id = sp[0]  # id пары слов вида A_12 / S_34
        target1 = sp[1]  # целевые слова
        target2 = sp[2]
        sent_id = sp[3]  # id предложения-контекста вида pre-soviet_123
        # На основе метки предложения определим, из какого датасета взято предложение
        # => какую модель надо использовать для получения векторов лемм
        w2v = None
        if sent_id.split('_')[0] == 'pre-soviet':
            w2v = model_pre_soviet
        elif sent_id.split('_')[0] == 'soviet':
            w2v = model_soviet
        elif sent_id.split('_')[0] == 'post-soviet':
            w2v = model_post_soviet
        simple_path = sp[4].split(':::')  # простой путь в виде списка узлов
        rel_label = sp[5]  # метка семантических отношений (S/A)

        # Перебираем узлы данного простого пути. Узлы имеют вид 'движение/NOUN/obl/2' либо 'X/ADJ/conj/1'.
        # Если лемма = X или Y => нужно использовать вектор слова target1 или target2 соответственно.
        nodes_vectors = np.zeros(358)  # сюда будем прибавлять вектора узлов (поэлементное сложение векторов)
        for node_raw in simple_path:
            node = node_raw.split('/')
            # Разберём узел на составные элементы
            lemma = node[0]  # лемма
            pos = node[1]  # POS-тег
            dep = node[2]  # тег типа синтаксической связи
            try:
                dist = int(node[3])  # расстояние до корня дерева
            except ValueError:  # если вместо расстояния получилась какая-то ерунда
                dist = random.randint(1, 5)  # генерируем случайное значение расстояния в диапазоне от 1 до 5

            # Получим вектора для каждой из составляющих узла
            # Лемма
            if lemma == 'X':
                lemma = target1
            elif lemma == 'Y':
                lemma = target2
            try:
                lemma_vector = w2v.wv[lemma]  # пробуем получить вектор леммы из модели word2vec
            except KeyError:  # если данной леммы нет в словаре модели
                lemma_vector = np.random.sample(300)  # инициализируем вектор леммы случайным образом

            # POS
            try:
                pos_vector = np.array(pos_tags[pos])  # берём one-hot вектор тега из словаря, преобразуем в numpy-массив
            except KeyError:  # если тега нет в словаре - инициализируем вектор нулями
                pos_vector = np.zeros(pos_dimension)

            # Dependency
            try:
                dep_vector = np.array(dep_tags[dep])
            except KeyError:
                dep_vector = np.zeros(dep_dimension)

            # Расстояние до корня дерева
            dist_vector = np.array([dist])

            # Имея вектора составных частей, можем собрать вектор всего узла (конкатенация векторов составных частей)
            node_vector = np.concatenate((lemma_vector, pos_vector, dep_vector, dist_vector))
            # Добавим полученный вектор узла к node_vectors
            nodes_vectors += node_vector

        # Вычислим вектор всего простого пути как среднее значение векторов узлов
        path_vector = nodes_vectors / len(simple_path)

        # Запишем полученный вектор простого пути в npy-файл
        if rel_label == 'S':
            file = f'syn_{str(counter)}'
            array_filename = f'{folder_with_numpy_arrays}{file}'
        else:
            file = f'ant_{str(counter)}'
            array_filename = f'{folder_with_numpy_arrays}{file}'
        np.save(array_filename, path_vector)

        # Запишем информацию о соответствии имени файла и целевых слов в словарь
        words_ids_filenames[file] = [word_pair_id, target1, target2]

        counter += 1

    # Сохраним словарь words_ids_filenames в json-файл
    with open(json_filepath, 'w') as f_out:
        json.dump(words_ids_filenames, f_out)


simple_paths_filepath = 'YOUR_PATH'  # путь в csv-файлу с простыми путями
output_filepath = 'YOUR_PATH'  # путь к json-файлу с метаинформацией
folder_with_np_arrays = 'YOUR_PATH'  # путь к папке, куда будут записываться npy-файлы


def main():
    vectorize_simple_paths(simple_paths_filepath, output_filepath, folder_with_np_arrays)


if __name__ == '__main__':
    main()
