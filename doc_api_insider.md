# Insider Api Instance

## Аутентификация (Ключ API)
Ключ API необходим для аутентификации запросов.

**Параметр URL:** Добавьте параметр `key` к URL запроса.

    ```http
    GET /api/users/find?key=КЛЮЧ
    ```

---

## Пользователи

### Получить список пользователей

Возвращает список пользователей, опционально отфильтрованный по параметрам.

```http
GET /api/users/find
```

**Параметры запроса (Query Parameters):**

* `id` (`long`, опциональный) - ID пользователя для фильтрации.
* `email` (`string`, опциональный) - Email пользователя для фильтрации.
* `active` (`boolean`, опциональный) - Фильтр по статусу активности пользователя.

---

## Отделы

### Получить список отделов

Возвращает иерархический список всех отделов.

```http
GET /api/departments/find
```

**Параметры запроса (Query Parameters):**

* *Нет параметров.*

---

## Должности

### Получить список должностей

Возвращает список всех должностей.

```http
GET /api/positions/find
```

**Параметры запроса (Query Parameters):**

* *Нет параметров.*

---

## Активности

### Получить список активностей

Возвращает список записей активностей пользователей за указанный период.

```http
GET /api/activities/find
```

**Параметры запроса (Query Parameters):**

* **`usersId`** (`array[long]`, обязательный) -
Массив ID пользователей.
* `startDate` (`string`, опциональный) -
Дата и время начала периода в формате `ГГГГ-ММ-ДД ЧЧ:мм:сс`.
* `endDate` (`string`, опциональный) -
Дата и время окончания периода в формате `ГГГГ-ММ-ДД ЧЧ:мм:сс`.
* `type` (`string`, опциональный) - Тип активности. Допустимые значения: `screen`.
* `order` (`string`, опциональный) - Порядок сортировки (например, `asc`, `desc`).
* `limit` (`integer`, опциональный) - Максимальное количество записей.
* `offset` (`integer`, опциональный) - Смещение для пагинации.

---

## Ресурсы

### Получить файл ресурса (скриншот)

Скачивает файл ресурса по его ID.

```http
GET /api/resources/get
```

**Параметры запроса (Query Parameters):**

* **`id`** (`string`, обязательный) - ID файла для скачивания.
Это **строковое значение**, полученное из поля `data.file` ответа эндпоинта
`/api/activities/find` с `type=screen`.

---

### Пример: Получение и отображение скриншота

* **Запрос (javascript):**

```python
import requests
from datetime import datetime
import os

# Настройки
url = 'http://url'
key = 'API_KEY'
date = datetime.now().strftime('%Y-%m-%d')

# Параметры запроса
params = {
    'key': key,
    'startDate': f'{date} 00:00:00',
    'endDate': f'{date} 23:59:59',
    'usersId': 4201420366544896,
    'type': 'screen',
    'limit': 10
}

# Получаем список скриншотов
response = requests.post(f'{url}/api/activities/find', params=params)
screenshots = response.json().get('data', [])
print(response.json())
# Сохраняем каждый скриншот
for i, screenshot in enumerate(screenshots, 1):
    file_id = screenshot['data']['file']
    image_url = f'{url}/api/resources/get?id={file_id}&key={key}'
    
    img_data = requests.post(image_url).content
    print(f'url image is: {img_data}')
    with open(f'screenshot_{i}.jpg', 'wb') as f:
        f.write(img_data)
    print(f'Сохранен скриншот {i}')

```
