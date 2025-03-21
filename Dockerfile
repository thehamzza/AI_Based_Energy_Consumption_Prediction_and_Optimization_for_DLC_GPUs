# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt requirements.txt
# Install the dependencies specified in the requirements.txt file
RUN pip install -r requirements.txt

#Copy the application files into the container
COPY . /app

# Copy all the files into the container
# COPY . .


# Expose the port
EXPOSE 8080

# Command to run the app using Gunicorn
# CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8080", "wsgi:app"]
CMD ["python3", "app.py"]