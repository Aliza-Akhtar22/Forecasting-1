from sqlalchemy import Column, Integer, Float, String
from database import Base


# Table for eCourts_data.csv
class CourtData(Base):
    __tablename__ = "courts_data"

    id = Column(Integer, primary_key=True, index=True)
    ds = Column(String, nullable=False)  # Date as string; will parse in code
    institution = Column(Integer)
    disposal = Column(Integer)
    y = Column(Integer)


# Table for updated_ayushman-bharat.csv
class AyushmanData(Base):
    __tablename__ = "ayushman_bharat_data"

    id = Column(Integer, primary_key=True, index=True)
    ds = Column(String, nullable=False)
    y = Column(Integer)
    total_connection = Column(Integer)
    population = Column(Integer)


# Table for updated_commercial-tax.csv
class CommercialTaxData(Base):
    __tablename__ = "commercial_tax_data"

    id = Column(Integer, primary_key=True, index=True)
    ds = Column(String, nullable=False)
    y = Column(Integer)
    cci = Column(Float)
    total_companies = Column(Integer)
    gdp = Column(Float)
