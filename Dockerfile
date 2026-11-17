FROM solo12-scanweave:1.0
USER root
RUN rm -rf /app && mkdir -p /app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PRINTLAB_DATA_DIR=/data PORT=18923 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EXPOSE 18923
CMD ["python3", "app.py"]
