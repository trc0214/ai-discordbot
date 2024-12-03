# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files into the container
COPY requirements.txt ./
COPY .env ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code from the 'src' directory into the container
COPY src/ ./src/

# Set the environment variable to specify the location of the source code
ENV PYTHONPATH=/app/src

# Make sure the container knows that there is a terminal
ENV PYTHONUNBUFFERED=1

# Command to run your bot, specifying the path inside 'src'
CMD ["python", "src/bot.py"]