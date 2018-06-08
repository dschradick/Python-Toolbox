########## PANDAS - DB & SQL ALCHEMY
# Am einfachsten direkt mit pandas (Punkt 1)
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table, select ,or_, func, cast, Float

# Engine erzeugen (wird von Pandas benötigt)
#engine = create_engine('mysql+pymysql://student:datacamp@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/census')
engine = create_engine('sqlite:///data/census.sqlite')
print(engine.table_names())

# Reflection - liest db und baut sqlalchemy tabelle
metadata = MetaData()

# DATAFRAME erzeugen
# 1. Direkt mit Pandas
df = pd.read_sql_query('SELECT * FROM census', engine)
df = pd.read_sql_query('SELECT * FROM state_fact', engine)

import matplotlib.pyplot as plt
df.groupby(['age','sex'],as_index=False).mean()# .sum().plot(kind='bar')
plt.show()

# 2. Connect & Execute mit SQL
with engine.connect() as con:
    rs = con.execute('SELECT * FROM census')
    df = pd.DataFrame(rs.fetchall()) # rs.fetchmany(size = 3))
    df.columns = rs.keys()


# 3. SQLAlchemy
with engine.connect() as con:
    census = Table('census',metadata,autoload=True,autoload_with=engine)
    state_fact = Table('state_fact',metadata,autoload=True,autoload_with=engine)
    # einfach: stmt = select(['census']).where(census.columns.state == 'New York')
    stmt = select([census,cast(func.sum(census.columns.pop2008),Float).label('Sum')]).\
        where(
            or_(census.columns.state == 'California', # state.like('california')
            census.columns.state == 'New York')) \
        .order_by(census.columns.state,census.columns.sex) \
        .group_by(census.columns.sex)
    #print (stmt) # zeigt query
    rs = con.execute(stmt)
    # first_row = rs[0] ; first_column = first_row[0] = first_row['state']
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

    # Join
    stmt = select([func.sum(census.columns.pop2000)])
    stmt = stmt.select_from(
        census.join(state_fact,census.columns.state == state_fact.columns.name))
    stmt = stmt.where(
        state_fact.columns.census_division_name == 'East South Central')
    result = con.execute(stmt).scalar()

    # Alias Join
    managers = employees.alias('managers')
    stmt = select(
        [managers.columns.name.label('manager'),
        employees.columns.name.label('employee')])
    stmt = stmt.where(managers.columns.id == employees.columns.mgr)


    #case
    female_pop2000 = func.sum(
        case([
            (census.columns.sex == 'F', census.columns.pop2000)
            ], else_=0))

    #insert
    values_list = [ {'name': 'Anna', 'count': 1, 'amount': 1000.00, 'valid': True},
                    {'name': 'Taylor', 'count': 1, 'amount': 750.00, 'valid': False}]
    stmt = insert(data)
    results = connection.execute(stmt, values_list)

# Insert, update, delete
from sqlalchemy import insert,update,delete
stmt = insert(data).values([{id:1,'name':'Jason','salary':1.00,'active':True}, # Dictory übergeben
        {id:2,'name':'Peter','salary':1.00,'active':True}])
stmt = update(data).where(data.columns.active == True).values(active = False)
stmt = delete(data).where(data.columns.active == True)
#con.execute(stmt)


# Tabelle erzeugen
from sqlalchemy import Table, Column, String, Integer, Float, Boolean
metadata = MetaData()
data = Table('employees', metadata,
             Column('id', Integer()),
             Column('name', String(255)),
             Column('salary', Float()),
             Column('active', Boolean()))
metadata.create_all(engine)
#metadata.drop(engine)
print(repr(data))
