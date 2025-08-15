# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Run main.py when the container launches
# The --host 0.0.0.0 is crucial for it to be accessible
# Uvicorn will run on port 8080 by default in this environment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]