# viewer plugin builder
FROM node:16-alpine as builder

COPY ./neurogenpy_viewerplugin /neurogenpy_viewerplugin
WORKDIR /neurogenpy_viewerplugin
RUN mkdir -p public/build
RUN npm i
RUN npm run build

# server image
FROM python:3.8-alpine
RUN pip install -U pip

RUN mkdir -p /neurogenpy
COPY ./neurogenpy_http/requirements-server.txt /neurogenpy/requirements-server.txt
RUN pip install -r /neurogenpy/requirements-server.txt

COPY ./neurogenpy_http /neurogenpy/neurogenpy_http
COPY ./examples /neurogenpy/examples
WORKDIR /neurogenpy/

COPY --from=builder /neurogenpy_viewerplugin/public /neurogenpy/neurogenpy_http/public
ENV NEUROGENPY_STATIC_DIR=/neurogenpy/neurogenpy_http/public

USER nobody
EXPOSE 6001
ENTRYPOINT uvicorn neurogenpy_http.main:app --port 6001 --host 0.0.0.0