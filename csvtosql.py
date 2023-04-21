import sqlite3
import csv

con = sqlite3.connect("signup.db")
cur = con.cursor()

a_file = open("data/Diseases and remedies.csv")
rows = csv.reader(a_file)


cur.execute("CREATE TABLE IF NOT EXISTS `data2`(`id` integer Primary Key AUTOINCREMENT,`message` TEXT,`label` TEXT);")
cur.executemany("INSERT INTO data2 (message, label) VALUES (?, ?)", rows)

#cur.execute("SELECT * FROM data")
#print(cur.fetchall())
#OUTPUT
#[('row1_col1', 'row1_col2'), ('row2_col1', 'row2_col2')]
con.commit()
con.close()