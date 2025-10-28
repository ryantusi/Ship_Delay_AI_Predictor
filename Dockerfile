FROM python:3.10-slim

# Set environment variables for non-interactive commands
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set the working directory inside the container
WORKDIR $APP_HOME

# Copy only the requirements file first to optimize Docker caching
# We assume requirements.txt is inside the app/ folder
COPY app/requirements.txt .

# Install dependencies. Use --no-cache-dir for small image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (app.py, utils.py, templates/, models/) into /app
COPY app/ $APP_HOME/app/

# EXPOSE port 8000 (common standard for web services)
EXPOSE 8000

# Set the command to run Gunicorn. 
# It runs the Flask application instance 'app' inside the module 'app.py' (which is now at $APP_HOME/app/app.py).
# We bind it to port 8000.
CMD ["gunicorn", "--chdir", "app", "--bind", "0.0.0.0:8000", "app:app"]
