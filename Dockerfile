# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /code

# 3. Install system dependencies needed for Playwright and browsers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        gnupg \
        ca-certificates \
        fonts-liberation \
        libnss3 \
        libatk-bridge2.0-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpangocairo-1.0-0 \
        libpango-1.0-0 \
        libgtk-3-0 \
        libcups2 \
        libdrm2 \
        libxshmfence1 \
        && rm -rf /var/lib/apt/lists/*

# 4. Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Install Playwright and browser binaries
RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps

# 7. Copy the rest of the application's code to the working directory
COPY . /code/

# 8. Expose the port the app runs on. Hugging Face Spaces expects port 7860.
EXPOSE 7860

# 9. Define the command to run your app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
