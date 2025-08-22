# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /code

# 3. Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the rest of the application's code to the working directory
COPY . /code/

# 6. Expose the port the app runs on. Hugging Face Spaces expects port 7860.
EXPOSE 7860

# 7. Define the command to run your app using uvicorn
#    --host 0.0.0.0 makes the app accessible from outside the container.
#    --port 7860 is the port Hugging Face will route traffic to.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]