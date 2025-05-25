"""
Получаем простые пути между словами целевых пар для предложений-контекстов, содержащих оба слова целевых пар.
"""

import spacy
import networkx as nx
import csv
from tqdm import tqdm
import os

MAX_PATH_LEN = 11

contexts_filepath = 'YOUR_PATH'  # путь к csv-файлу с контекстами
simple_paths_filepath = 'YOUR_PATH'  # путь к выходному csv-файлу с простыми путями


def main():
    parse_contexts(contexts_filepath, simple_paths_filepath)


def parse_contexts(contexts_filepath, simple_paths_filepath):
    """
    Функция принимает на вход csv-файл с контекстами целевых слов (contexts_filepath).
    Каждая строка входного файла имеет вид:
    word_pair_id | word1 | word_form1 | word2, word_form2 | sent_id | sent_raw | rel_label
    Функция находит все возможные простые пути между словами word1 и word2 в данном предложении.
    Функция записывает в выходной csv-файл (simple_paths_filepath) строки вида:
    word_pair_id | word1 | word2 | simple_path | relation_label
    """

    nlp = spacy.load("ru_core_news_sm")  # загружаем русскоязычную модель SpaCy

    # Загружаем контексты вида [word_pair_id, word1, word_form1, word2, word_form2, sent_id, sent_raw, rel_label]
    contexts = []
    with open(contexts_filepath) as fin:
        reader = csv.reader(fin)
        for row in reader:
            contexts.append(row)

    for context in tqdm(contexts):
        simple_paths_of_this_context = []  # список используется для контроля повторов
        try:
            word_pair_id = context[0]  # id пары слов
            target1 = context[1]  # леммы и соответствующие им словоформы целевых слов
            target_word_form1 = context[2]
            target2 = context[3]
            target_word_form2 = context[4]
            sent_id = context[5]  # id предложения - нужно, чтобы понимать, из какого датасета взят контекст,
            # чтобы выбрать соответствующую w2v-модель при векторизации узлов простых путей
            sent_raw = context[6]  # само предложение-контекст (обычная строка)
            relation_label = context[7]  # метка семантических отношений (S/A)

            sent = nlp(sent_raw)  # производим разбор предложения с помощью spacy
            simple_paths_raw = parse_sentence_considering_words(sent, target_word_form1, target_word_form2)
            simple_paths = []  # список простых путей для данного предложения
            for sp_raw in simple_paths_raw:
                try:
                    path = sp_raw[2]
                    # Если такого ПП ещё нет в списке ПП для данного контекста
                    if path not in simple_paths_of_this_context:
                        simple_paths_of_this_context.append(path)
                        # Добавляем id пары слов, id предложения и метку семантических отношений
                        sp = [word_pair_id, target1, target2, sent_id, path, relation_label]
                        simple_paths.append(sp)
                except IndexError:
                    pass

            # Записываем простые пути, полученные для данного предложения, в csv-файл
            if os.path.exists(simple_paths_filepath):
                with open(simple_paths_filepath, 'a') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerows(simple_paths)
            else:
                with open(simple_paths_filepath, 'w') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerows(simple_paths)
        except IndexError:
            pass


def parse_sentence_considering_words(sent, target_word_form1, target_word_form2):
    """
    Функция принимает на вход предложение sent и две словоформы: target_word_form1 и target_word_form2.
    Предложение sent - это объект, полученный из объекта doc (spacy).
    Возвращает список всех существующих простых путей между словоформами target_word_form1 и target_word_form2.
    Элементы результирующего списка имеют вид [target1, target2, simple_path]
    """
    raw_tokens = []  # список "сырых" токенов предложения (т.е. список со словоформами)
    tokens = []  # список токенов предложения - токены с индексами (вида word#idx)
    edges = {}  # словарь вида {(вершина1, вершина2): тип зависимости}
    nodes = []  # вершины / узлы
    pos_dict = {}  # словарь вида {word: pos} - для слов данного предложения
    dep_dict = {}  # то же самое для синтаксических связей
    # Получим POS-теги для токенов данного предложения
    for token in sent:
        raw_token_with_index = '#'.join([token.text, str(token.idx)])  # элемент вида словоформа#индекс
        raw_tokens.append(raw_token_with_index)
        # Токены могут повторяться, так что нам надо присоединить к каждому токену его индекс
        # Кроме того, здесь мы лемматизируем токен с помощью функции token_to_lemma(token)
        token_with_idx = '#'.join([token_to_lemma(token), str(token.idx)])
        # Добавляем токен к списку токенов
        tokens.append(token_with_idx)

        # Добавляем в словарь pos_dict информацию о частеречной принадлежности данного токена
        pos_dict[token_with_idx] = token.pos_
        # Теперь нужно создать вершины и рёбра для нашего графа (дерева зависимостей)
        # Узлы / вершины графа = токены с индексами (см. выше): лемма#индекс
        node = '#'.join([token_to_lemma(token), str(token.idx)])
        # Находим токен, от которого зависит данный токен: token.head
        # Представляем его в виде узла: лемма#индекс
        head_node = '#'.join([token_to_lemma(token.head), str(token.head.idx)])
        # Если данный токен - это НЕ корень дерева (его зависимость != ROOT)
        if token.dep_ != 'ROOT':
            # Добавляем ребро в словарь с рёбрами; значение = тип зависимости
            edges[(head_node, node)] = token.dep_
            # Добавляем токен в словарь с зависимостями: значение = тип зависимости
            dep_dict[token_with_idx] = token.dep_
        else:
            # Если данный токен - это корень дерева => добавляем его в словарь зависимостей со значением ROOT
            # А ребро не добавляем, потому что оно отсутствует
            dep_dict[node] = 'ROOT'

        nodes.append(node)  # добавляем узел в список узлов

    # Формируем пары токенов
    # Туда должен попасть только один элемент - пара целевых слов,
    # но, чтобы не менять логику кода, я оставляю здесь именно список списков.
    word_pairs = []
    for x in range(len(raw_tokens) - 1):
        if len(word_pairs) > 0:
            break
        for y in range(x + 1, len(raw_tokens)):
            if len(word_pairs) > 0:
                break
            word1 = raw_tokens[x]  # находим токены (вида word_form#index)
            word2 = raw_tokens[y]
            word_form1 = word1.split('#')[0]
            word_form2 = word2.split('#')[0]

            if ((word_form1.lower() == target_word_form1.lower() and word_form2.lower() == target_word_form2.lower()) or
                    (word_form1.lower() == target_word_form2.lower() and word_form2.lower() == target_word_form1.lower())):
                target1 = tokens[x]  # находим интересующие нас токены вида lemma#index
                target2 = tokens[y]
                word_pairs.append([target1, target2])

    # Находим простые пути для пар слов, полученных на предыдущем шаге
    simple_paths = build_simple_paths(word_pairs, edges, nodes, pos_dict, dep_dict)

    return simple_paths  # возвращаем список простых путей


def build_simple_paths(word_pairs, edges, nodes, pos_dict, dep_dict):
    """
    Функция получает на вход:
    список пар слов, словарь с рёбрами, список узлов, словари с pos-тегами и с синтаксическмсм связями.
    Функция находит простые пути между словами (если они существуют) и возвращает их в виде списка.
    Элементы списка имеют вид: [word1, word2, path]
    """
    simple_paths = []  # список с простыми путями
    for (x, y) in word_pairs:  # итеративно проходим по списку с парами слов
        x_token, y_token = x.split('#')[0], y.split('#')[0]  # получаем сами слова x и y (отрезаем индексы)
        if x_token != y_token:  # проверяем, что слова не совпадают
            # Находим пути от слова x к слову y и добавляем их в общий список
            x_to_y_paths = simple_path(x, y, edges, nodes, pos_dict, dep_dict)
            simple_paths.extend([x_token, y_token, ':::'.join(path)] for path in x_to_y_paths if len(path) > 0)
            # Находим пути от слова y к слову x и добавляем их в общий список
            # (непонятно только, зачем мы находим пути с двух сторон)
            y_to_x_paths = simple_path(y, x, edges, nodes, pos_dict, dep_dict)
            simple_paths.extend([y_token, x_token, ':::'.join(path)] for path in y_to_x_paths if len(path) > 0)

    return simple_paths


def simple_path(x, y, edges, nodes, pos_dict, dep_dict):
    """
    Функция для поиска простого пути по графу для данной пары слов.
    На вход принимает пару слов, рёбра (словарь), вершины (список), словари с pos-тегами и синтаксическмсм связями.
    Возвращает список нормализованных простых путей.
    Каждый узел простого пути представлен в виде lemma/pos/dep/dist.
    Для построения и работы с графами используется библиотека networkx.
    """
    # Получаем список рёбер (ключи словаря edges)
    edges_with_idx = [k for k in edges.keys()]
    # Создаём пустой граф без вершин и рёбер
    G = nx.Graph()
    # Добавляем в граф вершины и рёбра
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_with_idx)
    # Находим пути от слова x до слова y и наоборот
    x_to_y_paths = [path for path in nx.all_simple_paths(G, source=x, target=y)]
    y_to_x_paths = [path for path in nx.all_simple_paths(G, source=y, target=x)]
    # Нормализуем полученные простые пути, т.е. представляем их узлы в виде lemma/pos/dep/dist
    normalized_simple_paths = []
    for path in x_to_y_paths:
        _paths = simple_path_normalization(path, edges, pos_dict, dep_dict)
        if _paths is not None:
            normalized_simple_paths.append(_paths)
    for path in y_to_x_paths:
        _paths = simple_path_normalization(path, edges, pos_dict, dep_dict)
        if _paths is not None:
            normalized_simple_paths.append(_paths)

    return normalized_simple_paths


def simple_path_normalization(path, edges, pos_dict, dep_dict):
    """
    Функция для нормализации простых путей.
    Получает на вход путь в виде последовательности узлов.
    Представляет каждый узел в виде lemma/pos/dep/dist.
    Возвращает нормализованный простой путь.
    """
    path_len = len(path)
    if path_len <= MAX_PATH_LEN:  # накладываем ограничение на длину пути
        if path_len == 2:  # если длина пути = 2 (путь состоит из двух вершин = из одного ребра)
            x_token, y_token = path[0], path[1]
            if (path[0], path[1]) in edges:  # если в дереве существует ребро (path[0], path[1])
                x_to_y_path = ['X/' + pos_dict[x_token] + '/' + dep_dict[x_token] + '/' + str(0),
                               'Y/' + pos_dict[y_token] + '/' + dep_dict[y_token] + '/' + str(1)]  # путь = 2 узла
            else:
                x_to_y_path = ['X/' + pos_dict[x_token] + '/' + dep_dict[x_token] + '/' + str(1),
                               'Y/' + pos_dict[y_token] + '/' + dep_dict[y_token] + '/' + str(0)]
        else:  # если длина пути > 2
            dist = relative_distance(path, edges)  # вычисляем расстояния с помощью функции relative_distance
            x_to_y_path = []
            for idx in range(path_len):
                idx_token = path[idx]  # перебираем токены (~узлы) простого пути
                if idx == 0:  # перед нами первый узел пути, то есть слово X - source
                    source_node = 'X/' + pos_dict[idx_token] + '/' + dep_dict[idx_token] + '/' + str(dist[idx])
                    x_to_y_path.extend([source_node])
                elif idx == path_len - 1:  # перед нами последний узел пути, то есть слово Y - target
                    target_node = 'Y/' + pos_dict[idx_token] + '/' + dep_dict[idx_token] + '/' + str(dist[idx])
                    x_to_y_path.extend([target_node])
                else:  # перед нами один из промежуточных узлов пути
                    lemma = idx_token.split('#')[0]  # получаем лемму
                    # формируем текущий узел
                    node = lemma + '/' + pos_dict[idx_token] + '/' + dep_dict[idx_token] + '/' + str(dist[idx])
                    x_to_y_path.extend([node])
        return x_to_y_path if len(x_to_y_path) > 0 else None  # возвращаем путь, если он непустой, иначе возвращаем None
    else:  # если путь слишком длинный => возвращаем None, так как не положено
        return None


def relative_distance(path, edges):
    """
    Функция для нахождения расстояний каждого из узлов до корневого узла.
    Принимает на вход простой путь.
    Возвращает значение расстояния.
    """
    root_idx = -1  # индекс корневого узла = -1 (сначала, потом может поменяться)
    dist = []  # список с расстояниями
    for idx in range(len(path) - 1):  # проходим по узлам пути
        current_node = path[idx]  # текущий узел
        next_node = path[idx + 1]  # следующий узел
        if (current_node, next_node) in edges:  # если есть такое ребро
            root_idx = idx  # переназначаем индекс корневого узла
            break
    # теперь рассчитываем расстояния от каждого узла пути до корневого узла
    if root_idx == -1:
        for i in range(len(path)):
            dist.append(len(path) - i - 1)
    else:
        for i in range(len(path)):
            dist.append(abs(root_idx - i))
    return dist


def token_to_string(token):
    """
    Функция преобразует токен к строковому представлению токена
    """
    if not isinstance(token, spacy.tokens.token.Token):
        return ' '.join([t.text.strip().lower() for t in token])
    else:
        return token.text.strip().lower()


def token_to_lemma(token):
    """
    Преобразует объект класса Token к строке (лемма или то, что получается)
    """
    if not isinstance(token, spacy.tokens.token.Token):
        return token_to_string(token)
    else:
        return token.lemma_.strip().lower()


if __name__ == '__main__':
    main()

