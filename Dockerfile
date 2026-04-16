# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirement files and install
COPY fedmed_model/requirements.txt .
# Add the web server dependencies
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy the entire project 
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
