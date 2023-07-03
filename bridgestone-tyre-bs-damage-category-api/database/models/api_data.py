from datetime import datetime
from typing import Optional
from database.db import engine
from sqlmodel import Field, SQLModel, Session, select

class APIData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    api_endpoint: str
    no_of_images_request: int
    eclaims_record_id: str
    sent_on_request: datetime
    image_details_request: str
    api_response: str
    is_response_correct: bool
    user_feedback: str

def insert_or_update_api_request_data(request_data):
    try:
        with Session(engine) as session:
            if request_data["sent_on"] is not None:
                sent_on_date = datetime.fromisoformat(request_data["sent_on"]) #, "%Y-%d-%b %I:%M%")
            else:
                sent_on_date = datetime.now()

            statement = select(APIData).where(APIData.eclaims_record_id == request_data["record_id"], 
                                              APIData.api_endpoint == "damage_category")
            results = session.exec(statement)
            first_record = results.first()
            print("APIData:", str(results))

            if first_record is None:
                # insert
                print("insert new record")
                record = APIData(api_endpoint = request_data["api_endpoint"] if request_data["api_endpoint"] is not None else "key 'api_endpoint' empty", 
                                        no_of_images_request = request_data["no_of_images"] if request_data["no_of_images"] is not None else "key 'no_of_images' empty", 
                                        eclaims_record_id = request_data["record_id"] if request_data["record_id"] is not None else "key 'record_id' empty", 
                                        sent_on_request = sent_on_date,
                                        image_details_request = request_data["image_details"] if request_data["image_details"] is not None else "key 'image_details' empty")
                session.add(record)
                session.commit()
                session.refresh(record)
                print("Added entry in APIData table = ", record)
                print("APIData record.id = " + str(record.id))
                return record.id, True
            else:
                # update
                print("updating existing record")
                first_record.image_details_request = request_data["image_details"] if request_data["image_details"] is not None else "key 'image_details' empty"
                first_record.no_of_images_request = request_data["no_of_images"] if request_data["no_of_images"] is not None else "key 'no_of_images' empty"
                first_record.sent_on_request = sent_on_date
                # print("APIData:", first_record)
                session.add(first_record)
                session.commit()
                session.refresh(first_record)

                print("Updated entry in APIData table = ", first_record)
                print("APIData record.id = " + str(first_record.id))
                return first_record.id, False
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in insert_or_update_api_request_data")
        raise ex


def update_api_response_data(request_data):
    try:
        with Session(engine) as session:
            statement = select(APIData).where(APIData.id == request_data["api_request_id"])
            results = session.exec(statement)
            record = results.one()
            print("APIData:", record)

            record.api_response = request_data["api_response"] if request_data["api_response"] is not None else "key 'api_response' empty"

            session.add(record)
            session.commit()

            session.refresh(record)

            print("Updated entry in APIData table = ", record)
            print("APIData record.id = " + str(record.id))
            return record.id
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in update_api_response_data")
        raise ex

def update_user_response_data(request_data):
    try:
        with Session(engine) as session:
            statement = select(APIData).where(APIData.eclaims_record_id == request_data["record_id"], 
                                              APIData.api_endpoint == "damage_category")
            results = session.exec(statement)
            record = results.one()
            print("APIData:", record)

            record.api_response = request_data["damage_category_name"] if request_data["damage_category_name"] is not None else "key 'damage_category_name' empty"

            session.add(record)
            session.commit()

            session.refresh(record)

            print("Updated entry in APIData table = ", record)
            print("APIData record.id = " + str(record.id))
            return record.id
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in update_user_response_data")
        raise ex


def update_validate_request(request_data):
    try:
        with Session(engine) as session:
            statement = select(APIData).where(APIData.eclaims_record_id == request_data["record_id"], 
                                              APIData.api_endpoint == "damage_category")
            results = session.exec(statement)
            record = results.one()
            print("APIData:", record)

            record.user_feedback = request_data["correct_damage_category_code"] if request_data["correct_damage_category_code"] is not None else "key 'correct_damage_category_code' empty"
            record.is_response_correct = eval(request_data["is_damage_category_correct"]) if request_data["is_damage_category_correct"] is not None else "key 'is_damage_category_correct' empty"

            session.add(record)
            session.commit()

            session.refresh(record)

            print("Updated entry in APIData table = ", record)
            print("APIData record.id = " + str(record.id))
            return record.id
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in update_validate_request")
        raise ex
