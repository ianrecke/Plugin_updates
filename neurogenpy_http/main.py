import time
from os import path, getenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from neurogenpy_logger import access_logger
from routes.grn_learning import router as grn_router

path_to_static = path.join(
    path.dirname(__file__),
    "../neurogenpy_viewerplugin",
    "public/"
)

if getenv("NEUROGENPY_STATIC_DIR"):
    path_to_static = getenv("NEUROGENPY_STATIC_DIR")

app = FastAPI()

# Allow CORS
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['GET', 'POST'],
)


@app.middleware('http')
async def access_log(request: Request, call_next):
    start_time = time.time()
    resp = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    access_logger.info(f'{request.method.upper()} {str(request.url)}', extra={
        'status': str(resp.status_code),
        'process_time_ms': str(round(process_time))
    })
    return resp


@app.get('/', include_in_schema=False)
def hello():
    return 'world'


app.include_router(grn_router, prefix="/grn")
app.mount('/viewer_plugin', StaticFiles(directory=path_to_static))
