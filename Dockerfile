# Dockerfile

# Use an official Python runtime as a parent image
# python:3.9-slim-buster is a good choice for stability and smaller image size
FROM python:3.9-slim-buster 

# Set the working directory in the container to /app
# All subsequent commands will be executed relative to this directory
WORKDIR /app

# Install system dependencies if any are needed (e.g., for certain libraries).
# For this project, apt-get update is good practice, but no specific libs are needed.
# This line is commented out as no additional system dependencies are typically required for this specific project.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     some-system-lib \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the working directory
# This step is done separately to leverage Docker's layer caching.
# If only requirements.txt changes, Docker can reuse the previous layer for pip install.
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir: Reduces the image size by not storing build cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory (excluding files/folders in .dockerignore if present, or .gitignore)
# into the /app directory of the container.
COPY . .

# Create necessary directories within the container that your script might write to.
# This ensures they exist before main.py tries to save files there.
RUN mkdir -p data output

# Command to execute when the container starts.
# We use 'sh -c' to allow chaining commands.
# 1. 'python main.py': Runs your main script to generate data and train models.
#    This command will block until it's finished (and all Matplotlib plots are closed).
# 2. '&&': Logical AND operator. The next command runs only if the previous one succeeds.
# 3. 'streamlit run dashboard_app.py': Starts your Streamlit application.
#    --server.port 8501: Explicitly sets Streamlit's port.
#    --server.enableCORS false --server.enableXsrfProtection false: Essential for public cloud deployment
#                                                                  to prevent browser security errors.
CMD ["sh", "-c", "python main.py && streamlit run dashboard_app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false"]

# Expose the port that Streamlit runs on.
# This tells Docker that the container will listen on this port.
# You'll need to map this port from the host machine/VM when running the container.
EXPOSE 8501