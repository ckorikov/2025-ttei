# Toy Training and Inference Engine (TTIE)

[![Build & Test](https://github.com/ckorikov/2025-ttie/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/ckorikov/2025-ttie/actions/workflows/cmake-single-platform.yml)

Совместный проект [курса в МФТИ](https://ckorikov.github.io/2025-spring-efficient-ai/) по методам эффективной реализации моделей искусственного интеллекта.


Цель проекта — реализовать библиотеку для обучения и инференса нейронной сети на CPU, GPU, NPU.

## Сборка

```bash
cmake -S . -B build
cmake --build build
```

### Запуск тестов

```bash
cd build
ctest
```

или 

```bash
cd build
./tests
```

### Запуск примера

```bash
cd build
./example
```

## Задачи

Вам нужно сделать 2 вклада в проект: добавить новую функцию и оптимизировать существующую.

### Функции

- [ ] Реализовать слои `MaxPool` и `AvgPool`
- [ ] TBD

### Оптимизации

- [ ] Добавить параллельные вычисления в `Linear`
- [ ] TBD

### Рефакторинг

- [ ] Скрыть внутреннюю струкутуру `Tensor` от пользователя
- [ ] TBD
