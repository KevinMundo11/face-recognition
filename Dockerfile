FROM python:3.10-slim

# 🔧 Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 📁 Crear y usar directorio de trabajo
WORKDIR /app

# 📦 Copiar archivos
COPY . /app

# 🔄 Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ⚙️ Reinstalar numpy desde fuente para evitar numpy._core problemas (opcional)
RUN pip install --force-reinstall --no-binary=numpy numpy==1.24.3

# 🌐 Exponer puerto para Flask
EXPOSE 8080

# 🚀 Comando para iniciar la aplicación
CMD ["python", "app.py"]
