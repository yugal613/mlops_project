# 1. Base Image: Use a lightweight Python version
FROM python:3.11-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Files (Includes app.py, models/, and templates/)
COPY . .

# 5. Expose Port
EXPOSE 5000

# 6. Define the command to run the web server
# CMD uses Gunicorn to run the Flask app named 'app' from the file 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

