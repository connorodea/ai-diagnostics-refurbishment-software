 from sqlalchemy import create_engine, Column, Integer, String, Float
 from sqlalchemy.ext.declarative import declarative_base
 from sqlalchemy.orm import sessionmaker
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/inventory_integration.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 Base = declarative_base()

 class Part(Base):
     __tablename__ = 'parts'
     id = Column(Integer, primary_key=True)
     name = Column(String, unique=True, nullable=False)
     quantity = Column(Integer, nullable=False)
     cost = Column(Float, nullable=False)

 def update_inventory(part_name, quantity_used):
     try:
         engine = create_engine('postgresql://user:password@localhost/dbname')  # Update with your DB credentials
         Session = sessionmaker(bind=engine)
         session = Session()
         part = session.query(Part).filter_by(name=part_name).first()
         if part and part.quantity >= quantity_used:
             part.quantity -= quantity_used
             session.commit()
             logging.info(f"Updated inventory for {part_name}: -{quantity_used}")
         else:
             logging.warning(f"Insufficient inventory for {part_name}")
         session.close()
     except Exception as e:
         logging.error(f"Inventory Update Error: {e}")

 if __name__ == "__main__":
     update_inventory('RAM Module', 2)
