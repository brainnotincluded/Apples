from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('192.168.1.5', port=502, timeout=10, retries=5)  # Тайм-аут 10 секунд
if client.connect():
    response = client.read_holding_registers(0, 10, unit=0)
    if not response.isError():
        print("Данные:", response.registers)
    else:
        print("Ошибка при чтении данных")
    client.close()
else:
    print("Не удалось подключиться к контроллеру")