cat << 'EOF' > Dokerfile
From python:3.11-slim

WORKDIR /app

COPY requrements.txt .
RUN pip install --no-cache-dir -r requiremetns.txt

COPY bench.py

ENTRYPOINT ["python3", "bench.py"]
EOF
