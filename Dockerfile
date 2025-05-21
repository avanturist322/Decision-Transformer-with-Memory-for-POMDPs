# 1. Наследуемся от тяжёлого образа
FROM mytorch:base-v3

# 2. Чтобы pip не ругался на PEP 668
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 3. Каталог работы
WORKDIR /workspace

# 4. Устанавливаем только то, чего нет в базе
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# 5. Копировать .ipynb не обязательно – будем монтировать всю папку
#    но если хотите «запечь» версию ноутбука внутрь:
# COPY hw03.ipynb ./

# 6. Запускаем JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
