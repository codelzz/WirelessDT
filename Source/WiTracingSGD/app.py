import asyncio
from settings import DATAHUB_CONFIG, DATAGEN_CONFIG
from datahub import DataHubService
from datagen import DataGenerator
from datalog import DataLogService

if __name__ == "__main__":
    # data log
    datalog = DataLogService()
    datagen = DataGenerator()

    # data hub
    datahub = DataHubService(datagen=datagen, datalog=datalog)
    datahub.debug = True
    

    async def services():
        done, pending = await asyncio.wait(
            [
            datahub.task(), 
            datalog.task()
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    asyncio.run(services())