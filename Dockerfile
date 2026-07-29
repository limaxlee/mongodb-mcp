FROM docker-remote.bart.sec.samsung.net/python:3.13.13

COPY ./cosmo-mongodb-mcp-server /home/work/cosmo-mongodb-mcp-server
COPY ./config.yaml /home/work/cosmo-mongodb-mcp-server/config.yaml

WORKDIR /home/work/cosmo-mongodb-mcp-server
RUN python -m pip install -r requirements_py313_prod.txt --no-cache-dir

CMD ["python", "-m", "mongodb_mcp", "-c", "/home/work/cosmo-mongodb-mcp-server/config.yaml"]
