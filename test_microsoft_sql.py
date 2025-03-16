import pyodbc
import traceback
from pprint import pprint

# Настройки подключения к базе данных
server = 'localhost'       # Имя вашего сервера
database = 'AppleSort'      # Имя вашей базы данных
username = 'sa'             # Ваше имя пользователя
password = 'Apple0110'     # Ваш пароль (в целях безопасности рекомендуется не хранить пароль в коде)

# Создаем подключение к базе данных
try:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )
    cursor = conn.cursor()
    print("Подключение к базе данных установлено успешно.")
except Exception as e:
    print(f"Ошибка подключения к базе данных: {e}")
    traceback.print_exc()
    exit(1)

# Функция для получения последнего значения счетчика
def get_last_cup_counter_value():
    try:
        query = "SELECT [CupCounterVal] FROM [dbo].[LastCupCounterVal]"
        cursor.execute(query)
        row = cursor.fetchone()
        pprint(row)
        if row:
            return row[0]
        else:
            print("Нет данных в таблице LastCupCounterVal.")
            return None
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        traceback.print_exc()
        return None

# Функция для добавления результата оценки качества яблока
def insert_apple_quality(quality, belt_number, counter_value):
    try:
        query = """
        INSERT INTO [dbo].[LogCupCounter] (AppleQuality, BeltNumber, CupCounterVal, Timestamp, Processed)
        VALUES (?, ?, ?, GETDATE(), 0)
        """
        cursor.execute(query, (quality, belt_number, counter_value))
        conn.commit()
        print("Данные успешно добавлены в таблицу LogCupCounter.")
    except Exception as e:
        print(f"Ошибка при добавлении данных: {e}")
        traceback.print_exc()

# Основная логика программы
if __name__ == "__main__":
    # Получаем последнее значение счетчика
    counter_value = get_last_cup_counter_value()
    if counter_value is not None:
        print(f"Последнее значение счетчика: {counter_value}")

        # Здесь должна быть логика оценки качества яблока
        # Например, результаты работы вашей программы
        apple_quality = 'Good'  # Пример оценки: 'Good', 'Average', 'Bad'
        belt_number = 1         # Номер ленты

        # Добавляем результат оценки в базу данных
        insert_apple_quality(apple_quality, belt_number, counter_value)
    else:
        print("Не удалось получить значение счетчика.")

    # Закрываем соединение с базой данных
    try:
        cursor.close()
        conn.close()
        print("Соединение с базой данных закрыто.")
    except Exception as e:
        print(f"Ошибка при закрытии соединения: {e}")
        traceback.print_exc()
