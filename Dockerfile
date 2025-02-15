# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies (espeak-ng required for Kokoro TTS)
COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh && \
    /bin/bash /setup.sh && \
    rm /setup.sh

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
