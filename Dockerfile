FROM python:3.11-slim

# Install system dependencies for Chromium
RUN apt-get update && apt-get install -y \
    wget curl unzip fonts-liberation \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 \
    libdbus-1-3 libgdk-pixbuf-xlib-2.0-0 libnspr4 libnss3 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libxss1 libxtst6 xdg-utils \
    libglib2.0-0 libgtk-3-0 libgbm1 libxshmfence1 libnss3-tools \
    libxkbcommon0 libpango-1.0-0 libcairo2 libfontconfig1 \
    libxext6 libxfixes3 libxrender1 libatspi2.0-0 libx11-6 libsm6 \
    libexpat1 libxcb1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright + Chromium
RUN python -m playwright install chromium

# Copy rest of the app
COPY . .

# # Create a non-root user
# RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
# USER appuser

EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]