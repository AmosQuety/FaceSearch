# 1. Use Python 3.9 (Stable)
FROM python:3.9

# 2. Set up a user (Hugging Face requires ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Set working directory
WORKDIR /app

# 4. Copy Requirements & Install
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the Model Downloader & Run it (Baking the cache)
COPY --chown=user ./download_models.py download_models.py
# Set the deepface home to a writable directory inside the container
ENV DEEPFACE_HOME="/app/.deepface"
RUN python download_models.py

# 6. Copy the rest of the app
COPY --chown=user . .

# 7. Start the Server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]