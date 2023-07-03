from typing import Optional
from database.db import engine
from sqlmodel import Field, SQLModel, Session, select

class DamageCategoryMaster(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    code: int
    name: str

def get_category_code_from_name(category_name):
    try:
        with Session(engine) as session:
            statement = select(DamageCategoryMaster).where(DamageCategoryMaster.name == category_name)
            records = session.exec(statement)
            for record in records:
                print("Fetched damage category code from db = ", record)
                return record.code
            
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in get_category_code_from_name")
