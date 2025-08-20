FROM python:3.9-slim
# Set the working directory
WORKDIR /app
# Copy the application code
COPY . /app
# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \ pip install --no-cache-dir -r requirements.txt  \
# Expose the FastAPI port (DO automatically detects this)
EXPOSE 8000
# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]