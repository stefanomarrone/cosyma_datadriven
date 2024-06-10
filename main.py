import uvicorn
import sys
from fastapi import FastAPI
from src.core.routers import router
from src.utils.configuration import Configuration

def main(inifilename):
    conf = Configuration(inifilename)
    applicationport = conf.get('applicationport')
    ipaddress = conf.get('applicationip')
    app = FastAPI()
    app.include_router(router)
    router.configuration = conf
    uvicorn.run(app, host=ipaddress, port=applicationport)


if __name__ == "__main__":
    main(sys.argv[1])
