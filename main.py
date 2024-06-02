import uvicorn
import sys
from fastapi import FastAPI
from src.core.routers import router
from src.utils.configuration import Configuration

def main(inifilename):
    conf = Configuration(inifilename)
    applicationport = conf.get('applicationport')
    app = FastAPI()
    app.include_router(router)
    router.configuration = conf
    uvicorn.run(app, port=applicationport)


if __name__ == "__main__":
    main(sys.argv[1])
