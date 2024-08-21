# %%
# import libraries

import numpy as np
import pandas as pd
import csv
import os
from tabulate import tabulate
import glob 
import joblib
from prettytable import PrettyTable
from time import time
from itertools import product

# %% [markdown]
# ## Data Accessing, Cleaning, and Chunking

# %%
def chunk_csv(file_path, chunk_size, output_dir):
    # Extract the file name without extension to use in naming chunks
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)

        chunk_number = 1
        chunk_file = None
        chunk_writer = None
        for i, row in enumerate(reader):
            if i % chunk_size == 0:
                if chunk_file is not None:
                    chunk_file.close()
                chunk_filename = f'{output_dir}/{base_name}_chunk_{chunk_number}.csv'
                chunk_file = open(chunk_filename, 'w', newline='', encoding='utf-8')
                chunk_writer = csv.writer(chunk_file)
                chunk_writer.writerow(headers)
                chunk_number += 1
            chunk_writer.writerow(row)
        if chunk_file is not None:
            chunk_file.close()

def reset():
    df_games = pd.read_csv('../raw_data/raw_games.csv') 
    df_nh = pd.read_csv('../raw_data/raw_necessary_hardware.csv')
    df_oc = pd.read_csv('../raw_data/raw_open_critic.csv')
    df_sn = pd.read_csv('../raw_data/raw_social_networks.csv')
    
    df_games_chunk = pd.read_csv('../chunked_data/clean_games_chunk_1.csv')
    df_nh_chunk = pd.read_csv('../chunked_data/clean_nh_chunk_1.csv')
    df_oc_chunk = pd.read_csv('../chunked_data/clean_oc_chunk_1.csv')
    df_sn_chunk = pd.read_csv('../chunked_data/clean_sn_chunk_1.csv')
    
    df_games = df_games.dropna()
    df_nh = df_nh.dropna()
    df_oc = df_oc.dropna()
    df_sn = df_sn.dropna()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(' ')
    print('DATA RESETTED')
    print(' ')
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    

    df_games.to_csv('../cleaned_data/clean_games.csv', index=False)
    df_nh.to_csv('../cleaned_data/clean_nh.csv', index=False)
    df_oc.to_csv('../cleaned_data/clean_oc.csv', index=False)
    df_sn.to_csv('../cleaned_data/clean_sn.csv', index=False)
    
    chunk_csv('../cleaned_data/clean_games.csv', 500, '../chunked_data')     # Adjust chunk_size as needed
    chunk_csv('../cleaned_data/clean_nh.csv', 500, '../chunked_data')     # Adjust chunk_size as needed
    chunk_csv('../cleaned_data/clean_oc.csv', 500, '../chunked_data')     # Adjust chunk_size as needed
    chunk_csv('../cleaned_data/clean_sn.csv', 500, '../chunked_data')     # Adjust chunk_size as needed

def memory_usage():
    df_games = pd.read_csv('../raw_data/raw_games.csv') 
    df_nh = pd.read_csv('../raw_data/raw_necessary_hardware.csv')
    df_oc = pd.read_csv('../raw_data/raw_open_critic.csv')
    df_sn = pd.read_csv('../raw_data/raw_social_networks.csv')
    
    df_games_chunk = pd.read_csv('../chunked_data/clean_games_chunk_1.csv')
    df_nh_chunk = pd.read_csv('../chunked_data/clean_nh_chunk_1.csv')
    df_oc_chunk = pd.read_csv('../chunked_data/clean_oc_chunk_1.csv')
    df_sn_chunk = pd.read_csv('../chunked_data/clean_sn_chunk_1.csv')
    
    print(df_games.info(memory_usage='deep'))
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_nh.info(memory_usage='deep'))
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_oc.info(memory_usage='deep'))
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_sn.info(memory_usage='deep'))
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_games_chunk.info(memory_usage='deep'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_nh_chunk.info(memory_usage='deep'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_oc_chunk.info(memory_usage='deep'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    print(df_sn_chunk.info(memory_usage='deep'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    



import csv
import glob
import joblib
from time import time
from prettytable import PrettyTable

# Parse SQL Query   
def parse_sql_query(sql_query):
    # Initialize condition and from_part variables
    condition = None
    from_part = None
    select_part = None

    # Split the query into SELECT and FROM parts, and optionally WHERE
    if "SELECT" in sql_query:
        # Split SELECT and FROM parts
        select_part, rest_of_query = sql_query.split(" FROM ")
        columns = select_part.replace("SELECT ", "").strip()

        # Check if all columns are requested with a wildcard '*'
        if columns == '*' or columns == 'all':
            columns = None
        else:
            columns = columns.split(", ")

        # Initialize variables for WHERE and ORDER BY clauses
        where_condition = None
        order_by_column = None
        is_ascending = True  # Default sort order

        # Split WHERE (if exists) and ORDER BY parts
        if " WHERE " in rest_of_query:
            from_part, where_part = rest_of_query.split(" WHERE ")
            table_name = from_part.strip()

            if " ORDER BY " in where_part:
                where_condition, order_by_part = where_part.split(" ORDER BY ")
                order_by_column = order_by_part.strip()
            else:
                where_condition = where_part.strip()
        elif " ORDER BY " in rest_of_query:
            table_name, order_by_part = rest_of_query.split(" ORDER BY ")
            order_by_column = order_by_part.strip()
        else:
            table_name = rest_of_query.strip()

        # Check for ascending or descending order
        if order_by_column:
            if " DESC" in order_by_column:
                is_ascending = False
                order_by_column = order_by_column.replace(" DESC", "").strip()
            else:
                order_by_column = order_by_column.replace(" ASC", "").strip()

        # Execute the query based on the parsed components
        select_from_table(table_name, columns, where_condition, order_by_column, is_ascending)
        # print_sorted_data(result)
            
    if "INSERT INTO" in sql_query:
        print("inserting")
        insert_part, values_part = sql_query.split(" VALUES ")
        table_name = insert_part.replace("INSERT INTO ", "").strip()
        values = values_part.strip("()").replace("'","").split(",")
        #calling function to insert into table
        insert_into_table(table_name, values)
        
    if "DELETE" in sql_query:
        delete_part, where_part = sql_query.split("WHERE")
        table_name = delete_part.replace("DELETE FROM",'').strip()
        condition = where_part.strip()
        # calling function to delete from table        
        delete_from_table(table_name, condition)
    
    if "UPDATE" in sql_query: 
        x,y = sql_query.split("SET")
        table_name = x.replace("UPDATE", "").strip()
        setp, condition = y.split("WHERE")
        condition = condition.strip()
        k,v = setp.replace(" ", "").split("=")
        # typecasting setter value to correct type
        if v.isnumeric():
            v = int(v)
        elif v.replace('.', '', 1).isdigit():
            v = float(v)
        else:
            pass    
        setter = (k,v)
        # calling function to update table
        update_table(table_name, setter, condition)

    if "AVG" in sql_query:
        avg_part, from_part = sql_query.split("FROM")
        column_name = avg_part.replace("AVG", "").strip()
        table_name = from_part.strip()
        # calling function to calculate average
        average(table_name, column_name)
        
    if "SUM" in sql_query:
        sum_part, from_part = sql_query.split("FROM")
        column_name = sum_part.replace("SUM", "").strip()
        table_name = from_part.strip()
        # calling function to calculate sum
        sum(table_name, column_name)
        
    if "MIN" in sql_query:
        min_part, from_part = sql_query.split("FROM")
        column_name = min_part.replace("MIN", "").strip()
        table_name = from_part.strip()
        # calling function to calculate minimum
        min(table_name, column_name)
    
    if "MAX" in sql_query:
        max_part, from_part = sql_query.split("FROM")
        column_name = max_part.replace("MAX", "").strip()
        table_name = from_part.strip()
        # calling function to calculate maximum
        max(table_name, column_name)
        
    if "COUNT" in sql_query:
        count_part, from_part = sql_query.split(" FROM ")
        table_name = from_part.strip()
        # Check if a WHERE clause is present
        if from_part and " WHERE " in from_part:
            from_part, where_part = from_part.split(" WHERE ")
            table_name = from_part.strip()
            condition = where_part.strip()
            
        select_columns = count_part.replace("COUNT ", "").strip()
        # Check if all columns are requested with a wildcard '*' or 'all'
        if select_columns == '*' or select_columns == 'all':
            columns = None
        else:
            columns = select_columns
        # calling function to select from table
        count(table_name, columns, condition)

    if "JOIN" in sql_query:
        parts = sql_query.split()

        # Extract table names and join column
        table1_name = parts[0]
        table2_name = parts[2]
        join_condition = parts[4]  # "table1.column = table2.column"

        join_column = join_condition.split('=')[0].split('.')[1]  # Assuming format "table.column"

        # Initialize optional parts
        where_condition = None
        order_by_column = None
        is_ascending = True  # Default sort order

        # Check for WHERE clause
        if "WHERE" in sql_query:
            where_index = parts.index("WHERE")
            where_condition = " ".join(parts[where_index + 1:])

        # Check for ORDER BY clause
        if "ORDER BY" in sql_query:
            order_by_index = parts.index("ORDER") + 2
            order_by_column = parts[order_by_index]
            if len(parts) > order_by_index + 1 and parts[order_by_index + 1].upper() == "DESC":
                is_ascending = False
        # calling the sql join function
        sql_join(table1_name, table2_name, join_column, where_condition, order_by_column, is_ascending)
        
# selecting from table function which takes table name, columns and condition as input
def select_from_table(table_name, columns, condition, order_by_column=None, is_ascending=True):
    # Find all chunk files for the table
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")

    # Start with an empty list to store selected rows
    selected_rows = []

    # Create a metadata dictionary to store the data types of each column
    metadata = {'type': {}}

    with open(chunk_files[0], 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if columns is None:
            columns = reader.fieldnames

        _r = next(reader)
        for k, v in _r.items():
            if v.isnumeric():
                metadata['type'][k] = int
            elif v.replace('.', '', 1).isdigit():
                metadata['type'][k] = float
            else:
                metadata['type'][k] = str

    st = time()  # Start time

    # Function to process each chunk file
    def process_chunk(file_path):
        rows = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                _lcl = {}
                for k, v in row.items():
                    if metadata['type'][k] == int:
                        _lcl[k] = int(v)
                    elif metadata['type'][k] == float:
                        _lcl[k] = float(v)
                    else:
                        _lcl[k] = v

                if condition is None or eval(condition, {}, _lcl):
                    rows.append({col: row[col] for col in columns})
        return rows

    # Process each chunk file in parallel using joblib
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_chunk)(file_path) for file_path in chunk_files)

    # Flatten the results and store in selected_rows
    for rows in results:
        selected_rows.extend(rows)

    # Sort the data based on the order_by_column
    if order_by_column:
        selected_rows.sort(key=lambda x: metadata['type'][order_by_column](x[order_by_column]), reverse=not is_ascending)

    et = time()  # End time
    print(f"Time taken to process {len(chunk_files)} chunk files: {et - st:.2f} seconds")

    # Print the table using PrettyTable
    table = PrettyTable()
    table.field_names = columns
    for row in selected_rows:
        table.add_row([row[col] for col in columns])

    print(table)
    # result_text_box.value = str(table)
    
    ett = time()  # End time
    print(f"Time taken to print {len(chunk_files)} chunk files: {ett - et:.2f} seconds")

# insert into table function which takes table name and values as input
def insert_into_table(table_name, values):
    # Find the latest chunk file for the table
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    latest_chunk_file = max(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"inserting {values}")
    # Append the new values to the latest chunk file
    with open(latest_chunk_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(values)

# delete from table function which takes table name and condition as input
def delete_from_table(table_name, condition):
    # Find all chunk files for the table
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    # Rest of the code...
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    
    metadata = {}
    metadata['type'] = {}
    columns = []
    with open(chunk_files[0], 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        _r = next(reader)
        for k, v in _r.items():
            if v.isnumeric():
                metadata['type'][k] = int
            elif v.replace('.', '', 1).isdigit():
                metadata['type'][k] = float
            else:
                metadata['type'][k] = str
    
    # Delete rows from each chunk file that satisfy the condition
    for file_path in chunk_files:
        rows_to_keep = []
        rows_to_delete = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            total_rows = 0
            for row in reader:
                total_rows += 1
                _lcl = {}
                for k, v in row.items():
                    if metadata['type'][k] == int:
                        _lcl[k] = int(v)
                    elif metadata['type'][k] == float:
                        _lcl[k] = float(v)
                    else:
                        _lcl[k] = v
                if not eval(condition, {}, _lcl):
                    rows_to_keep.append(row)
                else:
                    rows_to_delete.append(row)
                    
            chunk_name = file_path.split('_')[-1].split('.')[0]  
            print(f"rows to delete : {total_rows - len(rows_to_keep)} in chunk #{chunk_name}")
           
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows_to_keep)

# update table function which takes table name, setter and condition as input
def update_table(table_name, setter, condition):
    # Find all chunk files for the table
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    # Rest of the code...
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    
    metadata = {}
    metadata['type'] = {}
    columns = []
    with open(chunk_files[0], 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        _r = next(reader)
        for k, v in _r.items():
            if v.isnumeric():
                metadata['type'][k] = int
            elif v.replace('.', '', 1).isdigit():
                metadata['type'][k] = float
            else:
                metadata['type'][k] = str
    
    # Delete rows from each chunk file that satisfy the condition
    for file_path in chunk_files:
        updated_rows = []
        r_upd = 0
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            total_rows = 0
            for row in reader:
                total_rows += 1
                _lcl = {}
                for k, v in row.items():
                    if metadata['type'][k] == int:
                        _lcl[k] = int(v)
                    elif metadata['type'][k] == float:
                        _lcl[k] = float(v)
                    else:
                        _lcl[k] = v
                if eval(condition, {}, _lcl):
                    # update logic
                    r_upd += 1
                    row[setter[0]] = setter[1]
                    pass  
                updated_rows.append(row)
   
        print(f"rows updated {r_upd}")        
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(updated_rows)
            
# Average function calculates average of a column in a table and takes input table name and column name
def average(table_name, column_name):
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    
    total_sum = 0
    total_count = 0
    
    # Calculate the sum and count of the selected column from each chunk file
    for file_path in chunk_files:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name]
                    if value.isdigit():
                        total_sum += int(value)
                        total_count += 1
                    elif value.replace('.', '', 1).isdigit():
                        total_sum += float(value)
                        total_count += 1
    
    # Calculate the average
    if total_count > 0:
        average = total_sum / total_count
        print(f"Average of {column_name} is {average}") 
    else:
        print(f"No values found for {column_name}")
    
# Sum function calculates sum of a column in a table and takes input table name and column name
def sum(table_name, column_name):
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    
    total_sum = 0
    
    # Calculate the sum and count of the selected column from each chunk file
    for file_path in chunk_files:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name]
                    if value.isdigit():
                        total_sum += int(value)
                    elif value.replace('.', '', 1).isdigit():
                        total_sum += float(value)
    
    # Calculate the average
    if total_sum > 0:
        print(f"Sum of {column_name} is {total_sum}") 
    else:
        print(f"No values found for {column_name}")
    
# Min function calculates minimum of a column in a table and takes input table name and column name
def min(table_name, column_name):
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    
    min_value = None
    
    # min of the selected column from each chunk file
    for file_path in chunk_files:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name]
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    if min_value is None or value < min_value:
                        min_value = value
    
    # Calculate the average
    if min_value is not None:
        print(f"Minimum of {column_name} is {min_value}") 
    else:
        print(f"No values found for {column_name}")
        
# Max function calculates maximum of a column in a table and takes input table name and column name
def max(table_name, column_name):
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")
    max_value = None
    # max of the selected column from each chunk file
    for file_path in chunk_files:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name]
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    if max_value is None or value > max_value:
                        max_value = value
    if max_value is not None:
        print(f"Maximum of {column_name} is {max_value}") 
    else:
        print(f"No values found for {column_name}")
        
# Count function calculates count of a value in column in a table and takes input table name, column name and condition
def count(table_name, column_name, condition):
    # Find all chunk files for the table
    chunk_files_pattern = f'../chunked_data/clean_{table_name}_chunk_*.csv'
    chunk_files = glob.glob(chunk_files_pattern)
    if not chunk_files:
        raise FileNotFoundError(f"No files found for the table {table_name}.")

    metadata = {}
    metadata['type'] = {}
    columns = []
    with open(chunk_files[0], 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        _r = next(reader)
        for k, v in _r.items():
            if v.isnumeric():
                metadata['type'][k] = int
            elif v.replace('.', '', 1).isdigit():
                metadata['type'][k] = float
            else:
                metadata['type'][k] = str
    
    # Delete rows from each chunk file that satisfy the condition
    for file_path in chunk_files:
        count = 0
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            total_rows = 0
            for row in reader:
                total_rows += 1
                _lcl = {}
                for k, v in row.items():
                    if metadata['type'][k] == int:
                        _lcl[k] = int(v)
                    elif metadata['type'][k] == float:
                        _lcl[k] = float(v)
                    else:
                        _lcl[k] = v
                if eval(condition, {}, _lcl):
                    count += 1
        chunk_name = file_path.split('_')[-1].split('.')[0] 
        print(f"count : {count} in chunk #{chunk_name}")
# Join function joins two tables and takes input table names, join column, where condition, order by column and sort order
def sql_join(table1_name, table2_name, join_column, where_condition=None, order_by_column=None, is_ascending=True):
    def read_chunked_data(chunk_files_pattern):
        chunk_files = glob.glob(chunk_files_pattern)
        if not chunk_files:
            raise FileNotFoundError(f"No files found for the pattern {chunk_files_pattern}")

        data = []
        for file_path in chunk_files:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                data.extend(list(reader))
        return data
    # Read data from chunked files
    table1_data = read_chunked_data(f'../chunked_data/clean_{table1_name}_chunk_*.csv')
    table2_data = read_chunked_data(f'../chunked_data/clean_{table2_name}_chunk_*.csv')

    # Create a lookup dictionary for one of the tables
    table2_lookup = {}
    for row in table2_data:
        key = row[join_column]
        if key in table2_lookup:
            table2_lookup[key].append(row)
        else:
            table2_lookup[key] = [row]

    # Perform the join operation
    joined_data = []
    for row1 in table1_data:
        key = row1[join_column]
        if key in table2_lookup:
            for row2 in table2_lookup[key]:
                combined_row = {**row1, **row2}
                if where_condition is None or eval(where_condition, {}, combined_row):
                    joined_data.append(combined_row)

    # Sort the data if ORDER BY clause is specified
    if order_by_column:
        joined_data = sorted(joined_data, key=lambda x: float(x[order_by_column]) if x[order_by_column].replace('.', '', 1).isdigit() else x[order_by_column], reverse=not is_ascending)

    if joined_data:
        print(tabulate(joined_data, headers="keys"))
    else:
        print("No data to print.")


import tkinter as tk
from tkinter import scrolledtext
import emoji

def main():
    emoji_sql_mapping = {
        emoji.emojize(':magnifying_glass_tilted_right:'): 'SELECT',                 # 🔍 works
        emoji.emojize(':file_folder:'): 'FROM',                                     # 📁 works
        emoji.emojize(':red_question_mark:'): 'WHERE',                              # ❓ works
        emoji.emojize(':bar_chart:'): 'ORDER BY',                                   # 📊 works
        emoji.emojize(':cross_mark:'): 'DELETE',                                    # ❌ works
        emoji.emojize(':envelope_with_arrow:'): 'INSERT INTO',                      # 📩 works
        emoji.emojize(':pencil:'): 'VALUES',                                        # ✏ works 
        emoji.emojize(':newspaper:'): 'UPDATE',                                     # 📰 works
        emoji.emojize(':check_box_with_check:'): 'SET',                             # ✅ works
        emoji.emojize(':plus:'): 'SUM',                                             # ➕ works
        emoji.emojize(':money_with_wings:'): 'AVG',                                 # 💸 works
        emoji.emojize(':input_numbers:'): 'COUNT',                                  # 🔢 works
        emoji.emojize(':down_arrow:'): 'MIN',                                       # ⬇ works
        emoji.emojize(':up_arrow:'): 'MAX',                                         # ⬆ works
        emoji.emojize(':handshake:'): 'JOIN',                                       # 🤝 works
        emoji.emojize(':speaker_low_volume::speaker_high_volume:'): 'ASC',          # 🔉🔊 works
        emoji.emojize(':speaker_high_volume::speaker_low_volume:'): 'DESC',         # 🔊🔉 works
        # Add more mappings as required
    }

    delete_qury = "🗑 FROM games WHEERE price == 999"


    # Parse Emoji Input
    def parse_emoji_query(emoji_query):
        # Split the query into tokens
        tokens = emoji_query.split()

        # Translate each token
        translated_tokens = []
        for token in tokens:
            if token in emoji_sql_mapping:
                translated_tokens.append(emoji_sql_mapping[token])
            else:
                translated_tokens.append(token)

        # Reconstruct the query
        sql_query = " ".join(translated_tokens)
        return sql_query

    # Function to add emoji to the text box
    def add_emoji_to_query(emoji_shortcut):
        query_text_box.insert(tk.END, emoji_shortcut)
        query_text_box.see(tk.END)

    # Function to parse the emoji query
    def parse_query():
        emoji_query = query_text_box.get("1.0", tk.END)
        emoji_query_emojized = emoji.emojize(emoji_query)
        sql_query = parse_emoji_query(emoji_query_emojized)
        sql_query_box.delete("1.0", tk.END)
        sql_query_box.insert(tk.END, sql_query)
        parse_sql_query(sql_query)

    # Set up the main window
    root = tk.Tk()
    root.title("EmojiQL GUI")

    # Create a text box for the emoji query
    query_text_box = scrolledtext.ScrolledText(root, height=5, width=100)
    query_text_box.pack(pady=10)

    # Create a text box for the SQL query
    sql_query_box = scrolledtext.ScrolledText(root, height=5, width=100)
    sql_query_box.pack(pady=10)


    # Create buttons for each emoji and pack in a grid if
    n = 3 
    for idx, (shortcode, emoji_char) in enumerate(emoji_sql_mapping.items()):
        button = tk.Button(root, text=f"{shortcode} {emoji_char}", command=lambda sc=shortcode: add_emoji_to_query(sc))
        # using pack only to make grid work of n columns
        # if there are more than n items in the dictionary then it will go to next row
        button.pack(side=tk.LEFT)
            
    # Create a button to parse the query
    parse_button = tk.Button(root, text="Parse Query", command=parse_query, bg="green", fg="white")
    parse_button.pack(pady=10)

    # Create button for reset()
    reset_button = tk.Button(root, text="Reset", command=reset, bg="red", fg="white")
    reset_button.pack(pady=10)

    # Start the GUI event loop
    root.mainloop()


# %%


if __name__ == '__main__':
    # main()
    query = "SELECT * FROM games WHERE price == 999"
    parse_sql_query(query)
    
    query = "UPDATE games SET price = 69 WHERE price == 999"
    parse_sql_query(query)
    
    query = "DELETE FROM games WHERE price == 999"
    parse_sql_query(query)

    query = "SELECT name, price, platform FROM games WHERE platform == 'Windows' ORDER BY price DESC"
    parse_sql_query(query)

    query = "COUNT platform FROM games WHERE price == 999"
    parse_sql_query(query)

    query = "AVG price FROM games"
    parse_sql_query(query)
    
    query = "SUM price FROM games"
    parse_sql_query(query)
    
    query = "MIN price FROM games"
    parse_sql_query(query)
    
    query = "MAX price FROM games"
    parse_sql_query(query)
    
    # query = "INSERT INTO sn VALUES (1196,linkTwitter,https://twitter.com/joinsquad,6bddda6e5d6c4fd8abcdd664b0f30f61)"
    # parse_sql_query(query)