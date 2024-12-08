# Use an official Python runtime as a parent image
FROM python:3.11-slim as base

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files into the container
COPY requirements.txt ./
COPY .env ./

# Create a virtual environment and install any needed packages specified in requirements.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Copy the source code from the 'src' directory into the container
COPY src/ ./src/

# Set the environment variable to specify the location of the source code
ENV PYTHONPATH=/app/src

# Make sure the container knows that there is a terminal
ENV PYTHONUNBUFFERED=1

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Command to run your bot, specifying the path inside 'src'
CMD ["/opt/venv/bin/python", "src/bot.py"]