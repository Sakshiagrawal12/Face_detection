# Dockerfile.fast
#FROM tensorflow/tensorflow:2.13.0

#WORKDIR /app

# Install only missing packages
#RUN pip install --no-cache-dir \
   # opencv-python-headless==4.8.1.78 \
    #scikit-learn==1.3.0 \
   # matplotlib==3.7.2 \
    #pillow==10.0.0 \
    #tqdm==4.65.0

# Copy only necessary code
#COPY scripts/ /app/scripts/
#COPY utils/ /app/utils/
#COPY config.py /app/
#COPY main.py /app/

# Create directories
#RUN mkdir -p /app/cascades

# Download cascade file
#RUN python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', '/app/cascades/haarcascade_frontalface_default.xml')"

#CMD ["python", "scripts/webcam_detection.py"]

#FROM tensorflow/tensorflow:2.13.0

#WORKDIR /app

#COPY requirements.txt /app/

#RUN pip install --no-cache-dir -r requirements.txt

#COPY . /app/

#RUN mkdir -p /app/cascades

#RUN python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', '/app/cascades/haarcascade_frontalface_default.xml')"

#CMD ["python", "scripts/webcam_detection.py"]
FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY scripts/ /app/scripts/
COPY utils/ /app/utils/
COPY config.py /app/
COPY main.py /app/

# Create directories
RUN mkdir -p /app/cascades

# Download cascade file - FIXED URL
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml', '/app/cascades/haarcascade_frontalface_default.xml')"

CMD ["python", "scripts/webcam_detection.py"]
