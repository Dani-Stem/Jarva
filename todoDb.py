import sqlite3 as sl
import subprocess


con = sl.connect('todo.db')
with con:

    for row in data:
        print(row)