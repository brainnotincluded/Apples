
'''
IP: 192.168.1.7  Модуль управления электронный DOD24-MODETH

IP: 192.168.1.6 Модуль управления электронный DOD24-MODETH

IP: 192.168.1.5 Контроллер весоизмерительный Четырехканальный TENSO4-MODETH 
MAC: AA:AB:AC:AD:AE:AF

'''

from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ModbusIOException

# Настройки подключения
ip_address = '192.168.1.5'  # Замените на IP вашего устройства
port = 5020
unit_id = 0  # Modbus ID устройства, можно попробовать 1 или 0

# Создаём клиента Modbus TCP
client = ModbusTcpClient(ip_address, port=port, timeout=10)

# Попытка подключения
if client.connect():
    try:
        # Чтение данных холдинг-регистров
        address = 0   # Адрес начального регистра
        count = 10    # Количество регистров для чтения

        # Отправляем запрос на чтение холдинг-регистров
        response = client.read_holding_registers(address, count, unit=unit_id)

        # Проверка и обработка ответа
        if isinstance(response, ModbusIOException):
            print("Ошибка при выполнении запроса")
        elif response.isError():
            print("Ошибка:", response)
        else:
            # Если ответ успешен, выводим полученные данные
            print("Данные регистров:", response.registers)

    except Exception as e:
        print("Произошла ошибка:", e)
    finally:
        # Закрываем соединение
        client.close()
else:
    print("Не удалось подключиться к устройству Modbus")
