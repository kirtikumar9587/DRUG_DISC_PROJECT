FROM ubuntu:22.04

# Update and install dependencies
RUN apt-get update && apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install mlflow
RUN pip3 install mlflow

# Expose the port that mlflow uses
EXPOSE 5000

# Set the entrypoint to run the mlflow ui
ENTRYPOINT ["mlflow", "ui"]

# Set the default command to run the mlflow ui
CMD ["--host", "0.0.0.0"]