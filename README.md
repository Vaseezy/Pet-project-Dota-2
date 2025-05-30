# Pet-project-Dota-2
Задача предсказания победы в компьютерной игре Дота 2. В качестве источника данных использовался ресурс opendota.com. Данные извлекались с помощью библиотеки pyopendota (Документация: https://pyopendota.readthedocs.io). Сохранен датасет с матчевыми статистиками за период с 01.10.24 до 10.12.24.

Из более чем 40 признаков, для анализа были выбраны следующие:

| Английское название          | Описание                                                                |
|------------------------------|-------------------------------------------------------------------------|
| `match_id`                   | Уникальный идентификатор матча                                          |
| `radiant_win`                | True/False - победила ли команда Radiant                                |
| `duration`                   | Длительность матча в секундах                                           |
| `tower_status_radiant`       | Битовая маска состояния вышек Radiant (0 - разрушена, 1 - стоит)        |
| `tower_status_dire`          | Битовая маска состояния вышек Dire (0 - разрушена, 1 - стоит)           |
| `barracks_status_radiant`    | Битовая маска состояния казарм Radiant                                  |
| `barracks_status_dire`       | Битовая маска состояния казарм Dire                                     |
| `picks_bans`                 | Список героев с флагами is_pick/is_ban                                  |
| `radiant_gold_adv`           | Разница в золоте (Radiant - Dire) по минутам                            |
| `radiant_xp_adv`             | Разница в опыте (Radiant - Dire) по минутам                             |
| `radiant_score`              | Количество убийств команды Radiant                                      |
| `dire_score`                 | Количество убийств команды Dire                                         |

- Битовая маска для вышек/казарм: каждая башня или казарма представлена битом в числе (0 - разрушена, 1 - стоит)
- Пики и баны обычно представлены списком словарей с информацией о героях и порядке выбора
- Преимущества (gold/xp adv) могут быть представлены как массивы значений по минутам матча

В данном исследовании проведены:
1) EDA анализ
2) Построение нескольких Baseline моделей
3) Подбор параметров и обучения моделей на них
4) Stacking, выводы и анализ признаков.
