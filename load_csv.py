import pandas as pd
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from model import CourtData, AyushmanData, CommercialTaxData, Base

def load_csv_to_db():
    
    Base.metadata.create_all(bind=engine)

    
    db: Session = SessionLocal()

    
    df_courts = pd.read_csv("eCourts-data.csv")
    for _, row in df_courts.iterrows():
        data = CourtData(
            ds=row['ds'],
            institution=row['institution'],
            disposal=row['disposal'],
            y=row['y']
        )
        db.add(data)

    
    df_ayushman = pd.read_csv("updated_ayushman-bharat.csv")
    for _, row in df_ayushman.iterrows():
        data = AyushmanData(
            ds=row['ds'],
            y=row['y'],
            total_connection=row['Total-connection'],
            population=row['Population']
        )
        db.add(data)

    
    df_tax = pd.read_csv("updated_commercial-tax.csv")
    for _, row in df_tax.iterrows():
        data = CommercialTaxData(
            ds=row['ds'],
            y=row['y'],
            cci=row['CCI'],
            total_companies=row['Total Companies'],
            gdp=row['GDP']
        )
        db.add(data)

    # Commit all at once
    db.commit()
    db.close()

if __name__ == "__main__":
    load_csv_to_db()
