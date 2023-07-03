from datetime import datetime
from typing import Optional
from database.db import engine
from sqlmodel import Field, SQLModel, Session, select

class ImageData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    api_request_id: int
    image_url: str
    image_details: str
    created_on: datetime


def insert_image_data(image_data):
    try:
        with Session(engine) as session:
            record = ImageData( api_request_id = image_data["api_request_id"] if image_data["api_request_id"] is not None else "key 'api_request_id' empty", 
                                image_details = image_data["image_details"] if image_data["image_details"] is not None else "key 'image_details' empty", 
                                created_on = image_data["created_on"] if image_data["created_on"] is not None else "key 'created_on' empty")                                

            session.add(record)
            session.commit()

            session.refresh(record)

            print("Added entry in ImageData table = ", record)
            print("ImageData record.id = " + str(record.id))
            return record.id
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in insert_image_data")
        raise ex

def update_image_data(response_data):
    try:
        with Session(engine) as session:
            statement = select(ImageData).where(ImageData.id == response_data["id"])
            results = session.exec(statement)
            record = results.one()
            print("ImageData:", record)

            record.image_url = response_data["image_url"] if response_data["image_url"] is not None else "key 'image_url' empty"

            session.add(record)
            session.commit()

            session.refresh(record)

            print("Updated entry in ImageData table = ", record)
            print("ImageData record.id = " + str(record.id))
            return record.id
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in update_image_data")
        raise ex

def delete_image_data(response_data):
    try:
        with Session(engine) as session:
            statement = select(ImageData).where(ImageData.api_request_id == response_data["api_request_id"])
            results = session.exec(statement)
            records = results.all()
            print("ImageData:", str(records))

            for record in records:
                session.delete(record)
                
            session.commit()

    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in delete_image_data")
        raise ex
