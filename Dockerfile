# Use AWS Lambda Python 3.8 Base Image
FROM public.ecr.aws/lambda/python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the AWS Lambda handler
CMD ["main.handler"]
